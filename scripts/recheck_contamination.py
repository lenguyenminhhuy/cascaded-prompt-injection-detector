"""E6c — contamination-controlled re-score.

Re-scores every E5 system on the eval set MINUS one source (default
openpromptinjection), post-hoc on saved logits. Motivation: E6a shows detection
is channel-structured — struq/direct DR~0.92 vs openpromptinjection/document
DR~0.05 — and channel==source 1:1 here, so the aggregate DR is dragged down by
the one genuinely-OOD document source. Removing it quantifies how much of the
low headline detection is "the unsolved document channel" rather than a property
of the detectors on the rest of the distribution.

Reports, for FULL eval vs eval-minus-<source>, side by side:
  * single-stage DR@1%FPR for each Stage-1 and for M2 (Stage-2)
  * cascade TPR@1%FPR, escalation e, leaked attacks (theta_safe frozen)

PAYLOAD HYGIENE (CLAUDE.md): reads only label/source + logit scores via the
tested load_rows loader; never input text. Emits only counts/rates/thresholds.

    PYTHONPATH=. python scripts/recheck_contamination.py \
        --exclude-source openpromptinjection \
        --out results/analysis/contamination_recheck.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.cascade_ablations import _cascade_masks  # noqa: E402
from scripts.score_stage1_logits import load_rows  # noqa: E402
from src.evaluation.metrics import detection_rate_at_fpr  # noqa: E402

FPR_TARGET = 0.01


def _dr(y, s, keep):
    d = detection_rate_at_fpr(y[keep], s[keep], (FPR_TARGET,))[str(FPR_TARGET)]
    return {"dr": round(d["dr"], 4), "threshold": round(float(d["threshold"]), 6),
            "achieved_fpr": round(d["achieved_fpr"], 4),
            "n_benign": int(d["n_benign"]), "n_injection": int(d["n_injection"])}


def _cascade(s1, s2, y, keep, theta_safe):
    ys, s1s, s2s = y[keep], s1[keep], s2[keep]
    m = _cascade_masks(s1s, s2s, ys, theta_safe, FPR_TARGET)
    is_att = ys == 1
    n_att = int(is_att.sum())
    return {
        "escalation_e": round(float(m["esc"].mean()), 4),
        "tpr": round(int(m["caught"].sum()) / n_att, 4),
        "leaked_autopass": int(m["leaked"].sum()),
        "leaked_share": round(int(m["leaked"].sum()) / n_att, 4),
        "theta2_escalated": round(m["theta2"], 6),
        "n_attacks": n_att,
    }


def parse_args():
    p = argparse.ArgumentParser(description="E6c contamination-controlled re-score")
    p.add_argument("--exclude-source", default="openpromptinjection")
    p.add_argument("--stage2-dir", default="results/stage2/mistral-7b-v0.1")
    p.add_argument("--stage2-name", default="mistral-7b (M2)")
    p.add_argument("--stage1", nargs="+",
                   default=["llama3.2-1b=results_stage1/stage1/llama3.2-1b",
                            "qwen2.5-1.5b=results_stage1/stage1/qwen2.5-1.5b"],
                   help="name=dir pairs; each dir holds eval_logits.jsonl")
    p.add_argument("--theta-safe", type=float, default=0.005)
    p.add_argument("--eval-split", default="data/eval_proposal/eval.jsonl")
    p.add_argument("--out", type=Path, default=ROOT / "results/analysis/contamination_recheck.json")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    eval_split = ROOT / a.eval_split
    # Stage-2 (shared label vector + sources)
    y, s2, srcs, _ = load_rows(Path(a.stage2_dir) / "eval_logits.jsonl", eval_split)
    y = np.asarray(y, int); s2 = np.asarray(s2, float)
    srcs = np.asarray([str(x) for x in srcs])
    keep_full = np.ones(len(y), bool)
    keep_ex = srcs != a.exclude_source
    n_removed = int((~keep_ex).sum())

    result = {
        "exclude_source": a.exclude_source,
        "n_total": int(len(y)), "n_removed": n_removed,
        "n_kept": int(keep_ex.sum()),
        "theta_safe": a.theta_safe, "fpr_target": FPR_TARGET,
        "single_stage": {}, "cascade": {},
        "note": "Positional join (logits carry no id), corroborated by per-source "
                "label purity. Thresholds are set on the benign distribution of the "
                "relevant (full or subset) set, so each column is self-consistent.",
    }

    # Single-stage M2
    result["single_stage"][a.stage2_name] = {
        "full": _dr(y, s2, keep_full), "excluded": _dr(y, s2, keep_ex)}

    for pair in a.stage1:
        name, d = pair.split("=", 1)
        ys, s1, srcs1, _ = load_rows(Path(d) / "eval_logits.jsonl", eval_split)
        ys = np.asarray(ys, int); s1 = np.asarray(s1, float)
        assert np.array_equal(ys, y), f"{name}: label vector differs from Stage-2"
        result["single_stage"][name] = {
            "full": _dr(ys, s1, keep_full), "excluded": _dr(ys, s1, keep_ex)}
        result["cascade"][f"{name}->M2"] = {
            "full": _cascade(s1, s2, y, keep_full, a.theta_safe),
            "excluded": _cascade(s1, s2, y, keep_ex, a.theta_safe)}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))

    print(f"\n=== E6c contamination re-score: exclude source='{a.exclude_source}' ===")
    print(f"total={result['n_total']}  removed={n_removed}  kept={result['n_kept']}\n")
    print(f"{'system':22s} {'DR@1%FPR full':>14s} {'DR@1%FPR -OPI':>14s} {'delta':>8s}")
    for name, d in result["single_stage"].items():
        f, e = d["full"]["dr"], d["excluded"]["dr"]
        print(f"{name:22s} {f:>14.4f} {e:>14.4f} {e-f:>+8.4f}")
    print(f"\n{'cascade':22s} {'TPR full':>9s} {'e full':>7s} | {'TPR -OPI':>9s} {'e -OPI':>7s} {'leaked-OPI':>11s}")
    for name, d in result["cascade"].items():
        f, e = d["full"], d["excluded"]
        print(f"{name:22s} {f['tpr']:>9.4f} {f['escalation_e']:>7.3f} | "
              f"{e['tpr']:>9.4f} {e['escalation_e']:>7.3f} {e['leaked_autopass']:>11d}")
    print(f"\nwrote -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
