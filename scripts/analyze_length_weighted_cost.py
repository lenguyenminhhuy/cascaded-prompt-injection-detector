"""Length-weighted cascade cost on the real evaluation traffic.

The headline cost table fixes one request length (512 tokens) and applies the
closed form 1 - (k1 + e*k2)/k2. Scoring is per-input with no padding to a fixed
length (see src/models/stage1.py::score_labels), so each request actually costs
what its own token count costs. This script re-does the accounting per input:

  baseline = sum_i k2(len_i)
  cascade  = sum_i k1(len_i) + sum_{i escalated} k2(len_i)

k(len) interpolates the measured latency curve (results/analysis/
latency_measured.json, four lengths) and is held flat outside it. Routing uses
the frozen theta_safe and the saved Stage-1 p_safe, so escalation is each
input's real decision, not an average.

Logits only + token counts; never prints text. Payload-safe.

Usage:
    python scripts/analyze_length_weighted_cost.py \
        --stage1-dir results_kaggle/stage1/qwen2.5-1.5b \
        --stage1-name qwen2.5-1.5b --stage1-e 0.397 \
        --out results/analysis/cost_length_weighted_qwen2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
# Anchors are derived from the latency file itself (see _curve) so a curve
# measured at extra lengths -- e.g. the 64-token point added by the short-input session --
# enters the interpolation instead of being silently ignored.
LENGTH_KEYS = None  # None => use every length present in the latency JSON
STAGE2 = "mistral-7b-v0.1"
THETA_SAFE = 0.005          # frozen on calibration; see leak_anatomy.json
SECONDS_PER_HOUR = 3600.0


def _curve(per_model: dict, name: str):
    """Measured (tokens, ms) anchors, sorted and deduplicated on token count.

    Targets below the prompt template's own length collapse to the same
    actual_prompt_tokens (a 32- and a 64-token target both render to ~86 tokens),
    which would put a duplicate x in np.interp. Such duplicates are averaged into
    one anchor so the curve stays single-valued.
    """
    by_len = per_model[name]["by_length"]
    keys = LENGTH_KEYS if LENGTH_KEYS is not None else sorted(by_len, key=int)
    agg: dict[int, list[float]] = {}
    for k in keys:
        rec = by_len[k]
        agg.setdefault(int(rec["actual_prompt_tokens"]), []).append(float(rec["median_ms"]))
    xs = np.array(sorted(agg), float)
    ys = np.array([sum(agg[int(x)]) / len(agg[int(x)]) for x in xs], float)
    return xs, ys


def _latency(per_model: dict, name: str, ntok: np.ndarray) -> np.ndarray:
    xs, ys = _curve(per_model, name)
    return np.interp(ntok, xs, ys)      # flat below xs[0] and above xs[-1]


def _token_counts(eval_path: Path, tokenizer_path: Path) -> np.ndarray:
    tok = Tokenizer.from_file(str(tokenizer_path))
    texts, sources = [], []
    with eval_path.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                texts.append(r.get("text") or r.get("input") or "")
                sources.append(r.get("source"))
    counts = np.array([len(e.ids) for e in tok.encode_batch(texts)], float)
    del texts                                   # drop payloads immediately
    return counts, np.asarray(sources, object)


def _p_safe(path: Path) -> np.ndarray:
    vals = []
    with path.open() as f:
        for line in f:
            if line.strip():
                vals.append(json.loads(line)["p_safe"])
    return np.asarray(vals, float)


def _usd_per_million(total_ms: float, n: int, usd_per_gpu_hour: float) -> float:
    mean_s = total_ms / n / 1000.0
    return mean_s * 1e6 / SECONDS_PER_HOUR * usd_per_gpu_hour


def evaluate(args) -> dict:
    global LENGTH_KEYS
    if args.length_keys:
        LENGTH_KEYS = [k.strip() for k in args.length_keys.split(",")]
    lat = json.load(open(ROOT / args.latency))
    per_model = lat["per_model"]
    usd = args.usd_per_gpu_hour or lat.get("gpu_hourly_usd_assumption", 1.20)

    ntok, sources = _token_counts(ROOT / args.eval_split,
                                  ROOT / args.stage1_dir / "adapter/tokenizer.json")
    p_safe = _p_safe(ROOT / args.stage1_dir / "eval_logits.jsonl")
    if len(p_safe) != len(ntok):
        raise SystemExit(f"row mismatch: {len(p_safe)} logits vs {len(ntok)} eval rows")

    escalated = p_safe < args.theta_safe
    k1 = _latency(per_model, args.stage1_name, ntok)
    k2 = _latency(per_model, STAGE2, ntok)
    n = len(ntok)

    baseline_ms = float(k2.sum())
    cascade_ms = float(k1.sum() + k2[escalated].sum())
    avoided_ms = float(k2[~escalated].sum())
    router_ms = float(k1.sum())

    # uniform-512 accounting, as reported in the headline cost table
    k1_512 = per_model[args.stage1_name]["by_length"]["512"]["median_ms"]
    k2_512 = per_model[STAGE2]["by_length"]["512"]["median_ms"]
    uniform_reduction = 1.0 - (k1_512 + args.stage1_e * k2_512) / k2_512

    by_source = {}
    for src in sorted(set(sources.tolist())):
        m = sources == src
        by_source[src] = {"n": int(m.sum()),
                          "median_tokens": float(np.median(ntok[m])),
                          "p90_tokens": float(np.percentile(ntok[m], 90))}

    return {
        "stage1_name": args.stage1_name,
        "stage2_name": STAGE2,
        "n_eval": n,
        "theta_safe": args.theta_safe,
        "gpu_hourly_usd_assumption": usd,
        "escalation_rate_measured": float(escalated.mean()),
        "escalation_rate_reported": args.stage1_e,
        "token_length": {
            "median": float(np.median(ntok)), "mean": float(ntok.mean()),
            "p10": float(np.percentile(ntok, 10)), "p90": float(np.percentile(ntok, 90)),
            "frac_below_shortest_measured_128": float((ntok < 128).mean()),
            "frac_at_or_above_512": float((ntok >= 512).mean()),
            "by_source": by_source,
        },
        "uniform_512_accounting": {
            "k1_ms": k1_512, "k2_ms": k2_512,
            "r": k1_512 / k2_512,
            "break_even_e": 1.0 - k1_512 / k2_512,
            "reduction": float(uniform_reduction),
            "usd_per_million_baseline": _usd_per_million(k2_512 * n, n, usd),
            "usd_per_million_cascade": _usd_per_million(
                (k1_512 + args.stage1_e * k2_512) * n, n, usd),
        },
        "length_weighted_accounting": {
            "mean_k1_ms": float(k1.mean()),
            "mean_k2_ms": float(k2.mean()),
            "r_effective": float(k1.mean() / k2.mean()),
            "break_even_e": float(1.0 - k1.mean() / k2.mean()),
            "reduction": float(1.0 - cascade_ms / baseline_ms),
            "mean_k2_on_escalated_ms": float(k2[escalated].mean()),
            "mean_k2_on_autopassed_ms": float(k2[~escalated].mean()),
            "share_of_stage2_work_on_escalated": float(
                k2[escalated].sum() / k2.sum()),
            "usd_per_million_baseline": _usd_per_million(baseline_ms, n, usd),
            "usd_per_million_cascade": _usd_per_million(cascade_ms, n, usd),
            "usd_per_million_saved": _usd_per_million(baseline_ms - cascade_ms, n, usd),
            "usd_per_million_stage2_avoided": _usd_per_million(avoided_ms, n, usd),
            "usd_per_million_router_cost": _usd_per_million(router_ms, n, usd),
        },
        "caveats": [
            "Latency is held flat below the shortest measured length (128 tokens); "
            f"{(ntok < 128).mean():.1%} of requests fall there, so the very-short "
            "regime is unmeasured and the sign of a near-zero result is not firm.",
            "Single-stream (batch=1) only. Batching amortises the fixed overhead "
            "that dominates at short lengths and should recover part of the gap.",
            "Token counts use the Stage-1 tokeniser; Stage-2's differs slightly.",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-dir", required=True)
    p.add_argument("--stage1-name", required=True)
    p.add_argument("--stage1-e", type=float, required=True,
                   help="reported escalation rate, for the uniform-512 comparison")
    p.add_argument("--eval-split", default="data/eval_proposal/eval.jsonl")
    p.add_argument("--latency", default="results/analysis/latency_measured.json")
    p.add_argument("--length-keys", default=None,
                   help="comma-separated subset of latency anchors to interpolate over "
                        "(default: every length present). Lets one session's curve be "
                        "re-read at another session's anchor set, isolating the effect of "
                        "adding an anchor from session-to-session drift.")
    p.add_argument("--theta-safe", type=float, default=THETA_SAFE)
    p.add_argument("--usd-per-gpu-hour", type=float, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    res = evaluate(args)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")

    lw, un = res["length_weighted_accounting"], res["uniform_512_accounting"]
    print(f"{res['stage1_name']}  n={res['n_eval']}  "
          f"escalation={res['escalation_rate_measured']:.4f}")
    print(f"  median request = {res['token_length']['median']:.0f} tokens; "
          f"{res['token_length']['frac_at_or_above_512']:.1%} >= 512")
    print(f"  uniform-512     reduction = {un['reduction']:+.1%}  "
          f"(r={un['r']:.3f}, break-even e={un['break_even_e']:.3f})")
    print(f"  length-weighted reduction = {lw['reduction']:+.1%}  "
          f"(r={lw['r_effective']:.3f}, break-even e={lw['break_even_e']:.3f})")
    print(f"  $/M: baseline={lw['usd_per_million_baseline']:.2f}  "
          f"cascade={lw['usd_per_million_cascade']:.2f}  "
          f"saved={lw['usd_per_million_saved']:.2f}")
    print(f"       stage-2 avoided={lw['usd_per_million_stage2_avoided']:.2f}  "
          f"router cost={lw['usd_per_million_router_cost']:.2f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
