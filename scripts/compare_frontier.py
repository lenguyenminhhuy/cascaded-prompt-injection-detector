"""Assemble the calibration frontier comparison — NO GPU, reads JSON only.

Separates two things the raw finding conflated:

  * **Rank ceiling** — the calibration-INVARIANT escalation floor from
    ``tail_diagnostics.json`` (exact score thresholds, zero-leak + zero benign-FP).
    This is what Stage-1's *discrimination* allows; no monotone recalibration and no
    threshold placement can beat it. (A uniform-grid frontier cannot even *reach* it
    when scores are squished near 0/1 — the optimal cut-point falls between grid
    steps — which is why we read the floor analytically, not off a grid sweep.)

  * **Deployment grid** — the frontier on the fixed config grid (theta_safe in
    [0.50,0.95], theta_unsafe in [0.05,0.50]) for raw / cal-fit-T / oracle-T scores.
    Raw squished scores are unresolvable by the fixed grid; calibration's real job is
    to spread them so the grid lands near the ceiling. The gap between cal-fit and
    oracle is the transfer problem (T fit on saturated cal does not carry to shift).

    PYTHONPATH=. python scripts/compare_frontier.py --calibration-dir results/analysis/calibration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logging import get_logger  # noqa: E402

log = get_logger("compare_frontier")

ILLUSTRATIVE_RATIO = 0.1  # k1/k2 for the illustrative cost-reduction column

# deployment subdir name under --calibration-dir -> human label
DEPLOY_DIRS = {
    "deploy_raw": "frontier_deploy_raw",        # raw scores, default deployment grid
    "deploy_calfit": "frontier_deploy_calfit",  # cal-fit T (realistic)
    "deploy_oracle": "frontier_deploy_oracle",  # oracle per-split T (best case)
}


def _cost_red(escalation: float, ratio: float = ILLUSTRATIVE_RATIO) -> float:
    """cost_reduction = 1 - k1/k2 - e  (see src/evaluation/cost.py)."""
    return 1.0 - ratio - escalation


def _op(summary_split: dict) -> dict:
    """Min-escalation zero-leak + FPR<=1% deployment operating point (or infeasible)."""
    op = summary_split.get("min_escalation_by_fpr_target", {}).get("fpr<=0.01", {})
    if op.get("feasible") is False:
        return {"feasible": False, "note": op.get("note")}
    ci = op.get("ci95", {}).get("escalation_rate", {})
    return {
        "feasible": True,
        "escalation_rate": op.get("escalation_rate"),
        "escalation_ci": [ci.get("lo"), ci.get("hi")] if ci else None,
        "cost_reduction_r0.1": op.get("cost_reduction_at_r0.1"),
    }


def _load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return {s["split"]: s for s in json.loads(path.read_text()).get("splits", [])}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble calibration frontier comparison")
    p.add_argument("--calibration-dir", type=Path,
                   default=ROOT / "results/analysis/calibration")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--figure", type=Path,
                   default=ROOT / "results/figures/frontier_comparison.png")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base = args.calibration_dir
    deploy = {k: _load_summary(base / d / "frontier_summary.json")
              for k, d in DEPLOY_DIRS.items()}
    tail = json.loads((base / "tail_diagnostics.json").read_text()) \
        if (base / "tail_diagnostics.json").exists() else {}

    splits = sorted(tail.keys() or {s for m in deploy.values() for s in m})
    out = {"illustrative_k1k2": ILLUSTRATIVE_RATIO, "splits": {}, "warnings": []}
    for split in splits:
        row = {}
        floor = tail.get(split, {}).get("rank_cost_floor", {}).get("escalation_floor")
        wall = tail.get(split, {}).get("confidently_safe_attacks", {}).get("count")
        row["ceiling"] = {  # calibration-invariant rank ceiling (from tail diagnostics)
            "escalation_floor": floor,
            "cost_reduction_r0.1": _cost_red(floor) if floor is not None else None,
            "confidently_safe_attacks": wall,
        }
        for key in DEPLOY_DIRS:
            if split in deploy.get(key, {}):
                row[key] = _op(deploy[key][split])
        # Sanity: no deployment point may beat the rank floor by more than the 1% FP slack.
        for key in DEPLOY_DIRS:
            d = row.get(key, {})
            if floor is not None and d.get("feasible") and d.get("escalation_rate") is not None:
                if d["escalation_rate"] < floor - 0.03:
                    msg = (f"{split}/{key}: deployment e={d['escalation_rate']:.3f} below rank "
                           f"floor {floor:.3f} — check (rank floor should be a lower bound).")
                    out["warnings"].append(msg); log.warning(msg)
        out["splits"][split] = row

    if args.out is None:
        args.out = base / "frontier_comparison.json"
    args.out.write_text(json.dumps(out, indent=2))
    _print_table(out, splits)
    if not args.no_plot:
        _plot(out, splits, args.figure)
    for w in out["warnings"]:
        print(f"  ! {w}")
    print(f"\nwrote -> {args.out}")
    return 0


def _print_table(out: dict, splits) -> None:
    def e(d):
        if not d:
            return "  -  "
        if "escalation_floor" in d:
            v = d["escalation_floor"]
            return f"{v*100:4.0f}%" if v is not None else "  -  "
        return f"{d['escalation_rate']*100:4.0f}%" if d.get("feasible") else " n/a "

    def c(d):
        if not d:
            return "   - "
        v = d.get("cost_reduction_r0.1")
        return f"{v*100:+4.0f}%" if (v is not None and d.get("feasible", True)) else "   - "

    print(f"\n{'split':<20} {'ceiling':>16} {'deploy_raw':>12} {'deploy_calfit':>15} "
          f"{'deploy_oracle':>15}")
    print(f"{'':<20} {'(rank floor)':>16} {'(realistic←)':>28} {'(best case)':>15}")
    print("-" * 82)
    for s in splits:
        r = out["splits"][s]
        wall = r["ceiling"].get("confidently_safe_attacks")
        tag = f"  [wall:{wall} safe-atk]" if wall else ""
        print(f"{s:<20} {e(r['ceiling'])+' / '+c(r['ceiling']):>16} "
              f"{e(r.get('deploy_raw'))+'/'+c(r.get('deploy_raw')):>12} "
              f"{e(r.get('deploy_calfit'))+'/'+c(r.get('deploy_calfit')):>15} "
              f"{e(r.get('deploy_oracle'))+'/'+c(r.get('deploy_oracle')):>15}{tag}")
    print("\n(escalation% / cost-reduction@k1k2=0.1;  n/a = no zero-leak+FPR<=1% point in grid)")


def _plot(out: dict, splits, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as ex:  # noqa: BLE001
        log.warning("matplotlib unavailable (%s) — skipping figure", ex)
        return
    keys = ["deploy_raw", "deploy_calfit", "deploy_oracle"]
    labels = ["deploy raw", "deploy cal-fit T", "deploy oracle T"]
    colors = ["#d64f4f", "#d98a1f", "#3f6bd6"]
    x = np.arange(len(splits)); w = 0.24
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for i, k in enumerate(keys):
        vals = [(out["splits"][s].get(k, {}).get("escalation_rate") or np.nan) * 100
                for s in splits]
        ax.bar(x + (i - 1) * w, vals, w, label=labels[i], color=colors[i], alpha=0.9)
    # rank-ceiling markers (from tail diagnostics)
    for j, s in enumerate(splits):
        fl = out["splits"][s]["ceiling"].get("escalation_floor")
        if fl is not None:
            ax.plot([x[j] - 1.6 * w, x[j] + 1.6 * w], [fl * 100, fl * 100],
                    color="#1f9d6b", lw=2.4, zorder=5,
                    label="rank ceiling (floor)" if j == 0 else None)
    ax.axhline(90, ls="--", color="#555", lw=1, label="break-even (k1/k2=0.1)")
    ax.set_xticks(x); ax.set_xticklabels(splits, fontsize=9)
    ax.set_ylabel("escalation for zero-leak & FPR<=1%  (%)")
    ax.set_ylim(0, 108); ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.set_title("Deployment grid vs. rank ceiling, by calibration", fontsize=11)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
