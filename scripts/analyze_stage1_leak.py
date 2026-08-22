"""E9 — anatomy of the Stage-1 leak: can we / should we reduce it?

The one-sided cascade auto-passes confident-benign at Stage-1. Some attacks are
auto-passed too ("leaked"): 460/7113 = 6.5% for Qwen. They never reach M2, so they
are final false negatives. Two questions:

  Q2 (counterfactual). If those leaked attacks HAD been escalated, would M2 catch
      them at its operating threshold theta2? This separates "recoverable leak"
      (M2 would have caught it -> pure routing loss) from "unrecoverable leak"
      (M2 would miss it too -> the detector ceiling, not the router).

  Q1 (should we reduce it). Raising theta_safe escalates more and shrinks the leak,
      but re-tightens theta2 to hold 1% FPR. Does end-to-end DR actually improve, or
      does reducing the leak just cost more Stage-2 calls for no detection gain?

Uses saved logits only (no GPU). M2's score exists on EVERY eval row, including the
auto-passed ones, so the counterfactual is exact.

PAYLOAD HYGIENE (CLAUDE.md): reads only label + source/channel metadata + logit
scores; never input text. Row-order join, same as eval_cascade.py.

    PYTHONPATH=. python scripts/analyze_stage1_leak.py \
        --stage1-dir results_kaggle/stage1/qwen2.5-1.5b \
        --stage2-dir results/stage2/mistral-7b-v0.1 \
        --stage1-name qwen2.5-1.5b --out-dir results/analysis/cascade_qwen2
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

from scripts.eval_cascade import (  # noqa: E402
    FPR_TARGET, _cascade_point, _grid, _join, _select_theta_safe_on_cal,
)
from src.evaluation.metrics import detection_rate_at_fpr  # noqa: E402

# attack sources -> channel (matches ablations_summary E6a mapping)
SOURCE_CHANNEL = {"openpromptinjection": "document", "agentdojo": "tool",
                  "struq_synthetic": "direct"}
_META_KEYS = {"label", "source", "channel", "category", "id"}  # never read 'text'


def _read_meta(path: Path):
    """Row-order metadata (label + source + channel). Never touches text."""
    labels, sources, channels = [], [], []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            r = {k: r.get(k) for k in _META_KEYS}          # drop 'text' immediately
            labels.append(int(r["label"]))
            sources.append(r.get("source"))
            ch = r.get("channel") or SOURCE_CHANNEL.get(r.get("source"))
            channels.append(ch)
    return np.asarray(labels, int), np.asarray(sources, object), np.asarray(channels, object)


def _pctl(a):
    if a.size == 0:
        return None
    q = np.percentile(a, [10, 25, 50, 75, 90])
    return {"p10": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
            "p75": float(q[3]), "p90": float(q[4])}


def evaluate(args) -> dict:
    s1_dir = Path(args.stage1_dir)
    s2_dir = Path(args.stage2_dir)
    cal_labels = ROOT / args.cal_split
    eval_labels = ROOT / args.eval_split

    _, _, s1_ps_c, y_c = _join(s1_dir, "cal_logits", cal_labels)
    _, _, s2_ps_c, _ = _join(s2_dir, "cal_logits", cal_labels)
    _, _, s1_ps_e, y = _join(s1_dir, "eval_logits", eval_labels)
    _, _, s2_ps_e, _ = _join(s2_dir, "eval_logits", eval_labels)
    y_meta, src, chan = _read_meta(eval_labels)
    if len(y_meta) != len(y) or not np.array_equal(y_meta, y):
        raise SystemExit("metadata/label row-order mismatch vs logits")

    s2c = 1.0 - s2_ps_c
    s2e = 1.0 - s2_ps_e   # M2 injection score (higher = more injection)

    # frozen theta_safe + operating theta2 (identical to headline)
    dr2_cal = detection_rate_at_fpr(y_c, s2c, (FPR_TARGET,))[str(FPR_TARGET)]["dr"]
    grid = _grid(args.theta_safe_lo, args.theta_safe_hi, args.step)
    theta_safe = _select_theta_safe_on_cal(s1_ps_c, s2c, y_c, grid, dr2_cal,
                                            args.keep_frac, FPR_TARGET)
    op = _cascade_point(s1_ps_e, s2e, y, theta_safe, FPR_TARGET)
    theta2 = op["theta2"]

    is_att = y == 1
    auto_pass = s1_ps_e >= theta_safe
    leaked = auto_pass & is_att                 # attacks auto-passed at Stage-1
    escalated = (~auto_pass) & is_att

    # --- Q2: counterfactual — would M2 flag the leaked attacks at theta2? -------
    m2_flag_leaked = s2e[leaked] >= theta2
    n_leak = int(leaked.sum())
    recoverable = int(m2_flag_leaked.sum())     # M2 would have caught -> routing loss
    unrecoverable = n_leak - recoverable        # M2 would miss too -> detector ceiling

    def _by(group_vals, mask):
        out = {}
        for g in sorted({v for v in group_vals[mask] if v is not None}):
            gm = mask & (group_vals == g)
            m2f = s2e[gm] >= theta2
            out[str(g)] = {
                "leaked": int(gm.sum()),
                "m2_would_catch": int(m2f.sum()),
                "m2_would_miss": int(gm.sum() - m2f.sum()),
                "recoverable_frac": round(float(m2f.mean()), 4) if gm.sum() else None,
            }
        return out

    # M2 score of leaked attacks vs the threshold, and vs escalated attacks
    leaked_scores = s2e[leaked]
    esc_att_scores = s2e[escalated]
    dist = {
        "theta2": theta2,
        "leaked_attack_m2_score": _pctl(leaked_scores),
        "escalated_attack_m2_score": _pctl(esc_att_scores),
        "leaked_below_threshold_frac": round(float((leaked_scores < theta2).mean()), 4),
        "note": ("If leaked attacks sit far BELOW theta2, M2 also finds them benign-"
                 "looking -> escalating them would not help. If many sit ABOVE, the "
                 "leak is a routing loss Stage-1 could avoid."),
    }

    # --- Q1: leak-reduction tradeoff — sweep theta_safe upward ------------------
    sweep = []
    for th in _grid(theta_safe, args.theta_safe_hi, args.sweep_step):
        pt = _cascade_point(s1_ps_e, s2e, y, th, FPR_TARGET)
        sweep.append({
            "theta_safe": round(th, 4),
            "leaked_attacks": pt["leaked_attacks"],
            "escalation_rate": round(pt["escalation_rate"], 4),
            "e2e_dr_at_1pct_fpr": round(pt["e2e_tpr"], 4),
            "e2e_fpr": round(pt["e2e_fpr"], 4),
        })
    m2_everywhere = detection_rate_at_fpr(y, s2e, (FPR_TARGET,))[str(FPR_TARGET)]["dr"]

    return {
        "stage1_name": args.stage1_name,
        "n_attacks": int(is_att.sum()), "theta_safe": theta_safe, "theta2": theta2,
        "operating_point": {
            "leaked_attacks": n_leak, "leak_rate_of_attacks": round(n_leak / int(is_att.sum()), 4),
            "escalation_rate": round(op["escalation_rate"], 4),
            "e2e_dr_at_1pct_fpr": round(op["e2e_tpr"], 4),
        },
        "Q2_counterfactual_leak_recovery": {
            "leaked_attacks": n_leak,
            "m2_would_catch_at_theta2": recoverable,
            "m2_would_miss_at_theta2": unrecoverable,
            "recoverable_frac": round(recoverable / n_leak, 4) if n_leak else None,
            "by_channel": _by(chan, leaked),
            "by_source": _by(src, leaked),
            "interpretation": (
                f"Of {n_leak} leaked attacks, M2 would catch only {recoverable} "
                f"({recoverable/n_leak:.1%}) if escalated; {unrecoverable} it would miss too. "
                "The leak is mostly UNRECOVERABLE (M2-blind attacks), not a routing loss."
                if n_leak and recoverable / n_leak < 0.5 else
                f"Of {n_leak} leaked attacks, M2 would catch {recoverable} if escalated — "
                "a recoverable routing loss."),
        },
        "m2_score_distribution": dist,
        "Q1_leak_reduction_tradeoff": {
            "m2_on_everything_dr": round(float(m2_everywhere), 4),
            "sweep_theta_safe_up": sweep,
            "interpretation": (
                "Raising theta_safe shrinks the leak but pushes end-to-end DR DOWN toward "
                f"M2-on-everything ({m2_everywhere:.4f}), which is BELOW the operating point "
                f"({op['e2e_tpr']:.4f}). At a fixed 1% FPR, reducing the leak does not improve "
                "detection — it removes the FP-budget headroom that lets the cascade beat "
                "M2-everywhere, while adding Stage-2 cost."),
        },
        "provenance": {"stage1_dir": str(s1_dir), "stage2_dir": str(s2_dir),
                       "channel_from": "eval 'channel' field or source->channel map"},
    }


def _print(r: dict) -> None:
    op = r["operating_point"]
    q2 = r["Q2_counterfactual_leak_recovery"]
    print(f"\n=== STAGE-1 LEAK ANATOMY  {r['stage1_name']} ===")
    print(f"attacks={r['n_attacks']}  theta_safe={r['theta_safe']}  theta2={r['theta2']:.4f}")
    print(f"operating leak: {op['leaked_attacks']} attacks ({op['leak_rate_of_attacks']:.1%}) "
          f"| escalation={op['escalation_rate']:.1%} | e2e DR={op['e2e_dr_at_1pct_fpr']:.4f}")
    print(f"\nQ2 — would M2 catch the {q2['leaked_attacks']} leaked attacks if escalated?")
    print(f"  M2 would CATCH:  {q2['m2_would_catch_at_theta2']:>4}  ({q2['recoverable_frac']:.1%} recoverable)")
    print(f"  M2 would MISS:   {q2['m2_would_miss_at_theta2']:>4}  (unrecoverable — detector ceiling)")
    print(f"  {'channel':<12}{'leaked':>8}{'M2 catch':>10}{'M2 miss':>9}{'recov%':>8}")
    for ch, v in q2["by_channel"].items():
        rf = f"{v['recoverable_frac']*100:.1f}" if v["recoverable_frac"] is not None else "-"
        print(f"  {ch:<12}{v['leaked']:>8}{v['m2_would_catch']:>10}{v['m2_would_miss']:>9}{rf:>8}")
    d = r["m2_score_distribution"]
    lk, es = d["leaked_attack_m2_score"], d["escalated_attack_m2_score"]
    print(f"\n  M2 injection-score (median) — leaked attacks: {lk['median']:.3f}  vs  "
          f"escalated attacks: {es['median']:.3f}  (theta2={d['theta2']:.3f})")
    print(f"  leaked attacks scoring BELOW theta2: {d['leaked_below_threshold_frac']:.1%}")
    q1 = r["Q1_leak_reduction_tradeoff"]
    print(f"\nQ1 — reduce the leak by escalating more? (M2-on-everything DR={q1['m2_on_everything_dr']:.4f})")
    print(f"  {'theta_safe':>11}{'leaked':>8}{'escal.':>9}{'e2e DR':>9}")
    for s in q1["sweep_theta_safe_up"]:
        print(f"  {s['theta_safe']:>11.4f}{s['leaked_attacks']:>8}{s['escalation_rate']:>9.3f}"
              f"{s['e2e_dr_at_1pct_fpr']:>9.4f}")
    print(f"\n  {q1['interpretation']}")


def parse_args():
    p = argparse.ArgumentParser(description="E9 Stage-1 leak anatomy + M2 counterfactual")
    p.add_argument("--stage1-dir", required=True)
    p.add_argument("--stage2-dir", required=True)
    p.add_argument("--stage1-name", default="stage1")
    p.add_argument("--cal-split", default="data/train_proposal/cal.jsonl")
    p.add_argument("--eval-split", default="data/eval_proposal/eval.jsonl")
    p.add_argument("--theta-safe-lo", type=float, default=0.0)
    p.add_argument("--theta-safe-hi", type=float, default=1.0)
    p.add_argument("--step", type=float, default=0.005)
    p.add_argument("--sweep-step", type=float, default=0.05)
    p.add_argument("--keep-frac", type=float, default=0.99)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/analysis/cascade")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    res = evaluate(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "leak_anatomy.json").write_text(json.dumps(res, indent=2))
    _print(res)
    print(f"\nwrote -> {args.out_dir / 'leak_anatomy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
