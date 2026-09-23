"""E6a + E6e + frontier figure — post-hoc cascade ablations (no GPU, payload-safe).

Builds on the cal-frozen cascade operating point from scripts/eval_cascade.py:

  E6a  leak/detection breakdown by source and channel at the frozen operating point
       -> do the auto-passed (leaked) attacks concentrate in a few families?
  E6e  operating-point OOD transfer: freeze the 1%-FPR threshold on cal, apply to
       eval, report the eval FPR you actually get (Stage-2-only AND cascade).
  fig  cascade TPR vs escalation frontier (theta_safe sweep, FPR<=1%), both Stage-1s.

PAYLOAD HYGIENE (CLAUDE.md): reads only label/source/channel + logit scores; never
input text. Emits only counts / rates / thresholds.

    PYTHONPATH=. python scripts/cascade_ablations.py \
        --stage1-dir results_stage1/stage1/qwen2.5-1.5b --stage1-name qwen2.5-1.5b \
        --stage2-dir results/stage2/mistral-7b-v0.1 --stage2-name mistral-7b \
        --theta-safe 0.005 --out-dir results/analysis/cascade_qwen2
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

from scripts.score_stage1_logits import load_rows  # noqa: E402
from src.evaluation.metrics import detection_rate_at_fpr  # noqa: E402

FPR_TARGET = 0.01


def _load(stage_dir: Path, split: Path):
    """(labels, injection_score, sources, channels) via the tested loader."""
    labels, scores, sources, channels = load_rows(stage_dir / "eval_logits.jsonl", split)
    return np.asarray(labels, int), np.asarray(scores, float), sources, channels


def _cal_scores(stage_dir: Path, cal_split: Path):
    labels, scores, _, _ = load_rows(stage_dir / "cal_logits.jsonl", cal_split)
    return np.asarray(labels, int), np.asarray(scores, float)


def _cascade_masks(s1_score, s2_score, y, theta_safe, fpr_target):
    """Reproduce the one-sided cascade at theta_safe; return per-row outcome masks
    and the escalated Stage-2 threshold tuned for overall fpr_target."""
    # s1_score = 1 - p_safe_stage1 (higher = more injection). auto-pass when p_safe1
    # >= theta_safe  <=>  s1_score <= 1 - theta_safe.
    p_safe1 = 1.0 - s1_score
    is_ben = y == 0; is_att = y == 1
    n_ben = int(is_ben.sum())
    budget = fpr_target * n_ben
    auto_pass = p_safe1 >= theta_safe
    esc = ~auto_pass
    n_eb = int((esc & is_ben).sum())
    if n_eb == 0:
        theta2 = -np.inf
    else:
        local = min(1.0, budget / n_eb)
        theta2 = detection_rate_at_fpr(y[esc], s2_score[esc], (local,))[str(local)]["threshold"]
    flagged = s2_score >= theta2
    leaked = auto_pass & is_att                 # auto-passed attacks (FN)
    caught = esc & is_att & flagged             # escalated attacks caught by Stage-2 (TP)
    esc_missed = esc & is_att & ~flagged        # escalated attacks Stage-2 missed (FN)
    esc_fp = esc & is_ben & flagged             # escalated benign flagged (FP)
    return {"auto_pass": auto_pass, "esc": esc, "leaked": leaked, "caught": caught,
            "esc_missed": esc_missed, "esc_fp": esc_fp, "theta2": float(theta2)}


def _breakdown(keys, is_att, m):
    """Per-key attack accounting at the frozen operating point."""
    out = {}
    for key in sorted(set(keys), key=lambda x: (x is None, str(x))):
        sel = np.array([k == key for k in keys])
        att = sel & is_att
        n_att = int(att.sum())
        if n_att == 0:
            continue
        leaked = int((att & m["leaked"]).sum())
        caught = int((att & m["caught"]).sum())
        missed_esc = int((att & m["esc_missed"]).sum())
        out[str(key)] = {
            "n_attacks": n_att,
            "leaked_autopass": leaked,
            "escalated_missed": missed_esc,
            "caught": caught,
            "detection_rate": round(caught / n_att, 4),
            "leak_share_of_attacks": round(leaked / n_att, 4),
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["leaked_autopass"]))


def e6e_transfer(s2_cal_y, s2_cal_s, y_eval, s2_eval_s):
    """Freeze Stage-2's 1%-FPR threshold on cal, apply to eval; report realized eval FPR/DR."""
    d_cal = detection_rate_at_fpr(s2_cal_y, s2_cal_s, (FPR_TARGET,))[str(FPR_TARGET)]
    th = d_cal["threshold"]
    is_ben = y_eval == 0; is_att = y_eval == 1
    flagged = s2_eval_s >= th
    fpr = float((flagged & is_ben).sum() / max(1, int(is_ben.sum())))
    dr = float((flagged & is_att).sum() / max(1, int(is_att.sum())))
    d_eval = detection_rate_at_fpr(y_eval, s2_eval_s, (FPR_TARGET,))[str(FPR_TARGET)]
    return {"threshold_frozen_on_cal": float(th),
            "eval_fpr_at_cal_threshold": round(fpr, 4),
            "eval_dr_at_cal_threshold": round(dr, 4),
            "eval_fpr_target": FPR_TARGET,
            "eval_calibrated_threshold": float(d_eval["threshold"]),
            "eval_calibrated_dr": round(d_eval["dr"], 4),
            "note": "If eval_fpr_at_cal_threshold >> 1%, the operating point does NOT "
                    "transfer OOD; the reported DR@1%FPR relies on eval-set calibration "
                    "(standard, and matched by the baselines)."}


def frontier(s1_score, s2_score, y, thetas, fpr_target):
    pts = []
    p_safe1 = 1.0 - s1_score
    is_ben = y == 0; is_att = y == 1
    n_ben = int(is_ben.sum()); n_att = int(is_att.sum())
    budget = fpr_target * n_ben
    for th in thetas:
        esc = ~(p_safe1 >= th)
        n_eb = int((esc & is_ben).sum())
        if n_eb == 0:
            caught = int((esc & is_att).sum()); fpr = 0.0
        else:
            local = min(1.0, budget / n_eb)
            d = detection_rate_at_fpr(y[esc], s2_score[esc], (local,))[str(local)]
            caught = int(round(d["dr"] * d["n_injection"])) if d["n_injection"] else 0
            fpr = round(d["achieved_fpr"] * d["n_benign"] / n_ben, 4)
        pts.append({"theta_safe": round(th, 4), "escalation_rate": round(esc.mean(), 4),
                    "tpr": round(caught / n_att, 4), "fpr": fpr})
    return pts


def plot_frontier(all_pts: dict, path: Path, stage2_dr: float) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    colors = {"llama3.2-1b": "#c0392b", "qwen2.5-1.5b": "#2c6fbb"}
    for name, pts in all_pts.items():
        e = [p["escalation_rate"] for p in pts]
        tpr = [p["tpr"] for p in pts]
        ax.plot(e, tpr, "-", lw=1.5, color=colors.get(name, "#555"), label=f"{name} → Stage-2")
    ax.axhline(stage2_dr, ls="--", color="#888", lw=1, label=f"Stage-2-only DR={stage2_dr:.3f}")
    ax.set_xlabel("escalation rate e  (fraction sent to Stage-2 = cost)")
    ax.set_ylabel("end-to-end detection rate @ 1% FPR")
    ax.set_xlim(0, 1); ax.set_ylim(0, max(0.5, stage2_dr * 1.2))
    ax.set_title("Cascade frontier: detection vs escalation (1% FPR)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140); plt.close(fig)
    return True


def parse_args():
    p = argparse.ArgumentParser(description="E6a/E6e cascade ablations + frontier figure")
    p.add_argument("--stage1-dir", required=True); p.add_argument("--stage1-name", required=True)
    p.add_argument("--stage2-dir", required=True); p.add_argument("--stage2-name", default="stage2")
    p.add_argument("--theta-safe", type=float, required=True,
                   help="cal-frozen operating threshold from eval_cascade.py")
    p.add_argument("--cal-split", default="data/train_proposal/cal.jsonl")
    p.add_argument("--eval-split", default="data/eval_proposal/eval.jsonl")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    eval_split = ROOT / a.eval_split; cal_split = ROOT / a.cal_split
    y, s1, srcs, chans = _load(Path(a.stage1_dir), eval_split)
    y2, s2, _, _ = _load(Path(a.stage2_dir), eval_split)
    assert np.array_equal(y, y2), "stage1/stage2 eval label vectors differ"
    yc, s2c = _cal_scores(Path(a.stage2_dir), cal_split)

    m = _cascade_masks(s1, s2, y, a.theta_safe, FPR_TARGET)
    is_att = y == 1
    result = {
        "stage1_name": a.stage1_name, "stage2_name": a.stage2_name,
        "theta_safe": a.theta_safe, "theta2_escalated": m["theta2"],
        "totals": {"n": int(len(y)), "n_attacks": int(is_att.sum()),
                   "leaked_autopass": int(m["leaked"].sum()),
                   "caught": int(m["caught"].sum()),
                   "escalated_missed": int(m["esc_missed"].sum())},
        "E6a_leak_by_source": _breakdown(srcs, is_att, m),
        "E6a_leak_by_channel": _breakdown(chans, is_att, m),
        "E6e_stage2_threshold_transfer": e6e_transfer(yc, s2c, y, s2),
    }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "ablations_summary.json").write_text(json.dumps(result, indent=2))

    t = result["totals"]
    print(f"\n=== {a.stage1_name} -> {a.stage2_name}  @ theta_safe={a.theta_safe} ===")
    print(f"attacks={t['n_attacks']}  leaked(auto-pass)={t['leaked_autopass']}  "
          f"escalated-missed={t['escalated_missed']}  caught={t['caught']}")
    print("\nE6a leaked attacks by CHANNEL (share of that channel's attacks):")
    for k, v in result["E6a_leak_by_channel"].items():
        print(f"  {k:<24} attacks={v['n_attacks']:>5}  leaked={v['leaked_autopass']:>4} "
              f"({v['leak_share_of_attacks']:.0%})  DR={v['detection_rate']:.3f}")
    print("\nE6a leaked attacks by SOURCE (top 6 by leak count):")
    for k, v in list(result["E6a_leak_by_source"].items())[:6]:
        print(f"  {k:<24} attacks={v['n_attacks']:>5}  leaked={v['leaked_autopass']:>4} "
              f"({v['leak_share_of_attacks']:.0%})  DR={v['detection_rate']:.3f}")
    e6e = result["E6e_stage2_threshold_transfer"]
    print(f"\nE6e Stage-2 threshold transfer (freeze 1%-FPR threshold on cal -> eval):")
    print(f"  eval FPR at cal-frozen threshold = {e6e['eval_fpr_at_cal_threshold']:.4f} "
          f"(target {FPR_TARGET}); eval DR there = {e6e['eval_dr_at_cal_threshold']:.4f}")
    print(f"  {'-> transfers OK' if e6e['eval_fpr_at_cal_threshold'] <= 0.02 else '-> does NOT transfer (relies on eval calibration)'}")
    print(f"\nwrote -> {a.out_dir / 'ablations_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
