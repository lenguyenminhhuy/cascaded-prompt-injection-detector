"""Cascade cost/escalation frontier — Stage-1 route() sweep with a MODELED Stage-2.

For each routing threshold pair (theta_safe, theta_unsafe) this sweeps the
two-sided band from ``src.pipeline.routing.route`` over Stage-1 ``p_safe`` scores
and reports, per split:

  * escalation rate ``e`` (fraction sent to Stage-2) -> the cost axis
  * ``leaked_attacks``   : attacks Stage-1 auto-passes (end-to-end false
                           negatives regardless of Stage-2)
  * ``auto_block_FP``    : benign Stage-1 auto-blocks (end-to-end false
                           positives regardless of Stage-2)
  * end-to-end recall / FPR under a MODELED Stage-2 on the escalated band
  * unit-invariant cost reduction vs Stage-2-only, per k1/k2 ratio

Cost is ``C(e) = k1 + e*k2`` (src/evaluation/cost.py). Because Stage-1 latency
k1 is unmeasured and there is no same-GPU (k1,k2) pair yet, the headline cost
axis is the *unit-invariant* reduction as a
function of the ratio r=k1/k2; one illustrative $ figure uses the measured
DataSentinel k2 ~= 1459 ms and k1 = r*k2.

LIMITATIONS (also written into every output header):
  1. Stage-2 is MODELED, not measured (M2 untrained; DataSentinel unusable at
     ~62-65% FPR and scored on a different dataset). Frontier is best-case
     (perfect) plus a parameterized (DR, FPR) setting.
  2. Cost is unit-invariant; absolute $ is ILLUSTRATIVE only (k1 unmeasured).
  3. Thresholds are PRE-CALIBRATION (E4) — this is the exploratory frontier,
     not the final calibrated operating point.
  4. Small n (253/366/523) — add paired-bootstrap CIs later.
  5. Logit provenance — current logits are from the REPO-bundled adapter; re-run
     scripts/score_split.py on the confirmed training adapter before quoting
     final numbers.

Payload-safe: reads only labels + logit scores, never input text.

    PYTHONPATH=. python scripts/sweep_thresholds.py \
        --logits-dir scratchpad/e2score2 --stage2-model perfect
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_stage1_logits import load_rows  # noqa: E402
from src.evaluation.cost import cascade_cost_summary, cost_reduction  # noqa: E402
from src.pipeline.routing import Thresholds, route  # noqa: E402
from src.utils.io import load_yaml  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)

# logit-file stem -> split jsonl providing the labels (mirrors rank_stage1.py)
DEFAULT_SPLIT_MAP = {
    "val": ROOT / "data/train_proposal/val.jsonl",
    "cal": ROOT / "data/train_proposal/cal.jsonl",
    "test_in_dist": ROOT / "data/splits/test_in_dist.jsonl",
    "test_cross_channel": ROOT / "data/splits/test_cross_channel.jsonl",
    "test_cross_domain": ROOT / "data/splits/test_cross_domain.jsonl",
}
DEFAULT_SPLITS = ["test_in_dist", "test_cross_channel", "test_cross_domain"]

# Measured Stage-2 (DataSentinel-7B KAD, batch=1) latency; illustrative $ only.
K2_MS_DATASENTINEL = 1459.0  # results/metrics/datasentinel_7b.json
ILLUSTRATIVE_RATIO = 0.1     # k1/k2 used for the single illustrative $ figure
PROPOSAL_E_TARGET = 0.25     # proposal working target: escalation <= 25%
FPR_TARGETS = [0.01, 0.05, 0.10]

LIMITATIONS = [
    "Stage-2 is MODELED, not measured (M2 untrained; DataSentinel unusable / wrong dataset).",
    "Cost is unit-invariant; absolute $ is ILLUSTRATIVE only (k1 unmeasured).",
    "Thresholds are PRE-CALIBRATION (E4) — exploratory frontier, not the final operating point.",
    "Small n (253/366/523) — add paired-bootstrap CIs later.",
    "Logits from the REPO-bundled adapter; re-score the confirmed training adapter for final numbers.",
]


def _grid(lo: float, hi: float, step: float) -> List[float]:
    """Inclusive grid, integer-stepped to avoid float drift."""
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 10) for i in range(n + 1)]


def _route_counts(p_safe: np.ndarray, is_attack: np.ndarray, is_benign: np.ndarray,
                  theta_safe: float, theta_unsafe: float) -> Dict[str, int]:
    """Vectorized equivalent of route() over the whole split (pass/block/escalate)."""
    pass_mask = p_safe >= theta_safe
    block_mask = p_safe <= theta_unsafe
    esc_mask = ~pass_mask & ~block_mask
    return {
        "escalated": int(esc_mask.sum()),
        "auto_pass": int(pass_mask.sum()),
        "auto_block": int(block_mask.sum()),
        "leaked_attacks": int((pass_mask & is_attack).sum()),
        "auto_block_attacks": int((block_mask & is_attack).sum()),
        "escalated_attacks": int((esc_mask & is_attack).sum()),
        "auto_block_FP": int((block_mask & is_benign).sum()),
        "escalated_benign": int((esc_mask & is_benign).sum()),
        "auto_pass_benign": int((pass_mask & is_benign).sum()),
    }


def _end_to_end(counts: Dict[str, int], n_attacks: int, n_benign: int,
                model: str, d2: float, f2: float):
    """End-to-end recall/FPR applying the modeled Stage-2 to the escalated band."""
    d2v, f2v = (1.0, 0.0) if model == "perfect" else (d2, f2)
    caught = counts["auto_block_attacks"] + d2v * counts["escalated_attacks"]
    flagged = counts["auto_block_FP"] + f2v * counts["escalated_benign"]
    recall = caught / n_attacks if n_attacks else float("nan")
    fpr = flagged / n_benign if n_benign else float("nan")
    return recall, fpr


def _cross_check_route(p_safe: np.ndarray, theta_safe: float, theta_unsafe: float,
                       counts: Dict[str, int]) -> None:
    """Assert the vectorized masks match the scalar route() on a small sample."""
    th = Thresholds(theta_safe=theta_safe, theta_unsafe=theta_unsafe)
    idx = range(0, len(p_safe), max(1, len(p_safe) // 50))  # ~<=50 probes
    esc = sum(1 for i in idx if route(float(p_safe[i]), th) == "escalate")
    esc_vec = sum(1 for i in idx
                  if not (p_safe[i] >= theta_safe or p_safe[i] <= theta_unsafe))
    assert esc == esc_vec, f"route() vs vectorized mismatch at ({theta_safe},{theta_unsafe})"


def sweep_split(split: str, logits_path: Path, split_path: Path, ts: dict,
                model: str, d2: float, f2: float, ratios: List[float]) -> dict:
    labels, scores, _sources, _channels = load_rows(logits_path, split_path)
    labels = np.asarray(labels, dtype=int)
    p_safe = 1.0 - np.asarray(scores, dtype=float)  # load_rows returns 1 - p_safe
    is_attack = labels == 1
    is_benign = labels == 0
    n = int(len(labels))
    n_attacks = int(is_attack.sum())
    n_benign = int(is_benign.sum())

    safe_grid = _grid(*ts["theta_safe_range"], ts["step"])
    unsafe_grid = _grid(*ts["theta_unsafe_range"], ts["step"])

    rows: List[dict] = []
    checked = False
    for theta_safe in safe_grid:
        for theta_unsafe in unsafe_grid:
            if theta_unsafe >= theta_safe:  # Thresholds requires strict <
                continue
            c = _route_counts(p_safe, is_attack, is_benign, theta_safe, theta_unsafe)
            if not checked:  # one-time correctness pin against scalar route()
                _cross_check_route(p_safe, theta_safe, theta_unsafe, c)
                checked = True
            e = c["escalated"] / n
            recall, fpr = _end_to_end(c, n_attacks, n_benign, model, d2, f2)
            rows.append({
                "theta_safe": theta_safe,
                "theta_unsafe": theta_unsafe,
                "escalation_rate": e,
                "residual_fn_rate": c["leaked_attacks"] / n_attacks if n_attacks else float("nan"),
                "leaked_attacks": c["leaked_attacks"],
                "auto_block_FP": c["auto_block_FP"],
                "e2e_recall": recall,
                "e2e_fpr": fpr,
                "cost_reduction_by_ratio": {
                    f"{r:g}": cost_reduction(r, 1.0, e) for r in ratios
                },
            })

    return {
        "split": split,
        "n": n, "n_attacks": n_attacks, "n_benign": n_benign,
        "stage2_model": model,
        "stage2_dr": None if model == "perfect" else d2,
        "stage2_fpr": None if model == "perfect" else f2,
        "k1k2_ratios": ratios,
        "grid": {"theta_safe_range": ts["theta_safe_range"],
                 "theta_unsafe_range": ts["theta_unsafe_range"], "step": ts["step"]},
        "limitations": LIMITATIONS,
        "rows": rows,
        # transient (popped before JSON dump) — raw arrays for the bootstrap
        "_arrays": {"p_safe": p_safe, "is_attack": is_attack, "is_benign": is_benign},
    }


def pareto_summary(result: dict) -> dict:
    """Min-escalation operating points subject to no attack leakage + FPR targets."""
    rows = result["rows"]
    no_leak = [r for r in rows if r["leaked_attacks"] == 0]

    def _op(r: dict) -> dict:
        e = r["escalation_rate"]
        illus = cascade_cost_summary(ILLUSTRATIVE_RATIO * K2_MS_DATASENTINEL,
                                     K2_MS_DATASENTINEL, e)
        return {
            "theta_safe": r["theta_safe"], "theta_unsafe": r["theta_unsafe"],
            "escalation_rate": e, "e2e_fpr": r["e2e_fpr"], "e2e_recall": r["e2e_recall"],
            "leaked_attacks": r["leaked_attacks"], "auto_block_FP": r["auto_block_FP"],
            "cost_reduction_at_r0.1": r["cost_reduction_by_ratio"].get(
                f"{ILLUSTRATIVE_RATIO:g}"),
            "meets_e_target_0.25": bool(e <= PROPOSAL_E_TARGET),
            "illustrative_usd": {
                "k1_over_k2": illus["k1_over_k2"],
                "cascade_is_cheaper": illus["cascade_is_cheaper"],
                "cost_reduction_pct": illus["cost_reduction_pct"],
                "expected_cost_usd_per_1m": illus["expected_cost_usd_per_1m"],
                "stage2_only_usd_per_1m": illus["stage2_only_usd_per_1m"],
            },
        }

    per_target = {}
    for t in FPR_TARGETS:
        feasible = [r for r in no_leak if r["e2e_fpr"] <= t]
        per_target[f"fpr<={t:g}"] = (
            _op(min(feasible, key=lambda r: r["escalation_rate"])) if feasible
            else {"feasible": False,
                  "note": f"no zero-leak point with e2e_fpr <= {t:g}"}
        )

    # Best FPR reachable while leaking zero attacks, and the escalation it costs:
    # this is the honest headline (min-escalation zero-leak can sit at FPR=1.0).
    best_fpr_zero_leak = (
        _op(min(no_leak, key=lambda r: (r["e2e_fpr"], r["escalation_rate"]))) if no_leak
        else {"feasible": False, "note": "no zero-leak operating point in the grid"}
    )
    # Leakage floor given the grid's widest safe threshold (max theta_safe).
    max_ts = max(r["theta_safe"] for r in rows)
    leak_floor = min(r["leaked_attacks"] for r in rows if r["theta_safe"] == max_ts)
    unsafe_lo = result["grid"]["theta_unsafe_range"][0]
    return {
        "split": result["split"], "n": result["n"],
        "n_attacks": result["n_attacks"], "n_benign": result["n_benign"],
        "stage2_model": result["stage2_model"],
        "leak_floor_at_max_theta_safe": {"theta_safe": max_ts, "leaked_attacks": leak_floor},
        "best_fpr_at_zero_leak": best_fpr_zero_leak,
        "min_escalation_by_fpr_target": per_target,
        "grid_note": (
            f"Reachable e2e_fpr is bounded below by the theta_unsafe floor "
            f"({unsafe_lo:g}): benign with p_safe <= {unsafe_lo:g} are auto-blocked "
            f"(counted as FP). Driving FPR lower requires escalating those benign "
            f"(theta_unsafe -> 0, escalation -> 1, cost saving -> 0). Widen with "
            f"--theta-unsafe-range 0.0 <hi> to expose the full tail."),
        "verdict": _verdict(best_fpr_zero_leak, per_target, leak_floor),
    }


def _verdict(best_zero_leak: dict, per_target: dict, leak_floor: int) -> str:
    if best_zero_leak.get("feasible") is False:
        return (f"NO zero-leak point in grid: >= {leak_floor} attack(s) auto-pass even at "
                f"the widest safe threshold. Cascade cannot avoid leaking attacks here.")
    tightest = per_target.get(f"fpr<={FPR_TARGETS[0]:g}", {})
    if tightest.get("feasible") is False:
        e = best_zero_leak["escalation_rate"]
        f = best_zero_leak["e2e_fpr"]
        return (f"Even at zero attack-leak the best end-to-end FPR is {f:.2f}, and it "
                f"costs escalation e={e:.2f} (no point reaches FPR<={FPR_TARGETS[0]:g}). "
                f"Cascade buys little cost saving at a safe FPR.")
    e = tightest["escalation_rate"]
    ok = tightest["meets_e_target_0.25"]
    return (f"Zero-leak + FPR<={FPR_TARGETS[0]:g} reachable at e={e:.2f} "
            f"({'MEETS' if ok else 'MISSES'} the e<=0.25 target).")


def _ci95(vals: np.ndarray) -> dict:
    """2.5 / 50 / 97.5 percentiles of a bootstrap distribution."""
    lo, med, hi = np.percentile(vals, [2.5, 50.0, 97.5])
    return {"lo": float(lo), "median": float(med), "hi": float(hi)}


def _bootstrap_point(arrays: dict, theta_safe: float, theta_unsafe: float,
                     model: str, d2: float, f2: float, n_boot: int,
                     rng: "np.random.Generator") -> dict:
    """Resample rows with replacement; recompute metrics at FIXED thresholds."""
    p_safe = arrays["p_safe"]; is_attack = arrays["is_attack"]; is_benign = arrays["is_benign"]
    n = len(p_safe)
    e_s, fpr_s, rec_s, leak_s = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ps, ia, ib = p_safe[idx], is_attack[idx], is_benign[idx]
        c = _route_counts(ps, ia, ib, theta_safe, theta_unsafe)
        na, nb = int(ia.sum()), int(ib.sum())
        recall, fpr = _end_to_end(c, na, nb, model, d2, f2)
        e_s.append(c["escalated"] / n)
        fpr_s.append(fpr); rec_s.append(recall)
        leak_s.append(c["leaked_attacks"])
    return {
        "n_boot": n_boot,
        "escalation_rate": _ci95(np.asarray(e_s)),
        "e2e_fpr": _ci95(np.asarray(fpr_s)),
        "e2e_recall": _ci95(np.asarray(rec_s)),
        "leaked_attacks": _ci95(np.asarray(leak_s)),
    }


def attach_bootstrap_cis(summary: dict, arrays: dict, model: str, d2: float,
                         f2: float, n_boot: int, seed: int) -> None:
    """Add 95% bootstrap CIs to each feasible operating point in the summary.

    CIs are computed at the point-estimate operating thresholds; they capture
    sampling noise in the metrics, NOT threshold-selection variance (the chosen
    (theta_safe, theta_unsafe) is held fixed across resamples).
    """
    rng = np.random.default_rng(seed)
    ops = [summary.get("best_fpr_at_zero_leak")]
    ops += list(summary.get("min_escalation_by_fpr_target", {}).values())
    for op in ops:
        if isinstance(op, dict) and "theta_safe" in op:
            op["ci95"] = _bootstrap_point(
                arrays, op["theta_safe"], op["theta_unsafe"],
                model, d2, f2, n_boot, rng)
    summary["bootstrap"] = {
        "n_boot": n_boot, "seed": seed,
        "note": ("95% CIs at the point-estimate operating thresholds "
                 "(paired row resampling); excludes threshold-selection variance."),
    }


def _write_csv(result: dict, path: Path, ratios: List[float]) -> None:
    ratio_cols = [f"cost_reduction_r{r:g}" for r in ratios]
    cols = ["theta_safe", "theta_unsafe", "escalation_rate", "residual_fn_rate",
            "leaked_attacks", "auto_block_FP", "e2e_recall", "e2e_fpr", *ratio_cols]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in result["rows"]:
            w.writerow([r["theta_safe"], r["theta_unsafe"], r["escalation_rate"],
                        r["residual_fn_rate"], r["leaked_attacks"], r["auto_block_FP"],
                        r["e2e_recall"], r["e2e_fpr"],
                        *[r["cost_reduction_by_ratio"][f"{rr:g}"] for rr in ratios]])


def _maybe_plot(result: dict, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    rows = result["rows"]
    e = [r["escalation_rate"] for r in rows]
    fpr = [r["e2e_fpr"] for r in rows]
    leaked0 = [r["leaked_attacks"] == 0 for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].scatter([x for x, k in zip(e, leaked0) if not k],
                  [y for y, k in zip(fpr, leaked0) if not k],
                  s=6, c="#c0392b", alpha=.35, label="attacks leak")
    ax[0].scatter([x for x, k in zip(e, leaked0) if k],
                  [y for y, k in zip(fpr, leaked0) if k],
                  s=6, c="#1f8a4c", alpha=.5, label="zero leak")
    ax[0].set_xlabel("escalation rate e (cost)"); ax[0].set_ylabel("end-to-end FPR")
    ax[0].set_title(f"{result['split']} — quality vs cost ({result['stage2_model']} Stage-2)")
    ax[0].legend(fontsize=8)
    cr = [r["cost_reduction_by_ratio"].get(f"{ILLUSTRATIVE_RATIO:g}") for r in rows]
    ax[1].scatter(fpr, cr, s=6, c="#3f4bb0", alpha=.35)
    ax[1].axhline(0, color="#888", lw=.8)
    ax[1].set_xlabel("end-to-end FPR"); ax[1].set_ylabel(f"cost reduction (r={ILLUSTRATIVE_RATIO})")
    ax[1].set_title("cost reduction vs FPR")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cascade cost/escalation frontier sweep")
    p.add_argument("--logits-dir", type=Path, default=ROOT / "scratchpad/e2score2",
                   help="dir of <split>_logits.jsonl (Stage-1 p_safe dumps)")
    p.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    p.add_argument("--config", type=Path, default=ROOT / "configs/evaluation.yaml")
    p.add_argument("--stage2-model", choices=["perfect", "param"], default="perfect")
    p.add_argument("--stage2-dr", type=float, default=0.95,
                   help="modeled Stage-2 detection rate on escalated attacks (param mode)")
    p.add_argument("--stage2-fpr", type=float, default=0.05,
                   help="modeled Stage-2 FPR on escalated benign (param mode)")
    p.add_argument("--k1k2-ratios", default="0.02,0.05,0.1,0.2,0.3",
                   help="comma-separated k1/k2 ratios for unit-invariant cost reduction")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/analysis/cost_frontier")
    p.add_argument("--no-plots", action="store_true")
    # Grid overrides: default None -> use configs/evaluation.yaml threshold_sweep.
    # Widen to e.g. --theta-unsafe-range 0.0 0.5 --theta-safe-range 0.5 1.0 to
    # expose the full frontier tail (escalate the low-p_safe benign, e->1).
    p.add_argument("--theta-safe-range", nargs=2, type=float, default=None,
                   metavar=("LO", "HI"))
    p.add_argument("--theta-unsafe-range", nargs=2, type=float, default=None,
                   metavar=("LO", "HI"))
    p.add_argument("--step", type=float, default=None, help="grid step (default config)")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="N paired row-resamples for 95%% CIs on operating points (0=off)")
    p.add_argument("--bootstrap-seed", type=int, default=0,
                   help="RNG seed for reproducible bootstrap CIs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(str(args.config))
    ratios = [float(x) for x in args.k1k2_ratios.split(",") if x.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the effective grid: config defaults, overridable per-axis via CLI.
    cfg_ts = cfg["threshold_sweep"]
    ts = {
        "theta_safe_range": list(args.theta_safe_range) if args.theta_safe_range
        else cfg_ts["theta_safe_range"],
        "theta_unsafe_range": list(args.theta_unsafe_range) if args.theta_unsafe_range
        else cfg_ts["theta_unsafe_range"],
        "step": args.step if args.step is not None else cfg_ts["step"],
    }

    summaries = []
    for split in args.splits:
        split_path = DEFAULT_SPLIT_MAP.get(split)
        logits_path = args.logits_dir / f"{split}_logits.jsonl"
        if split_path is None or not logits_path.exists() or not split_path.exists():
            log.warning("skip %s (logits or split file missing: %s)", split, logits_path)
            continue
        log.info("sweeping %s (%s Stage-2)", split, args.stage2_model)
        result = sweep_split(split, logits_path, split_path, ts,
                             args.stage2_model, args.stage2_dr, args.stage2_fpr, ratios)
        arrays = result.pop("_arrays")  # transient — never serialized
        (args.out_dir / f"{split}.json").write_text(json.dumps(result, indent=2))
        _write_csv(result, args.out_dir / f"{split}.csv", ratios)
        if not args.no_plots:
            if _maybe_plot(result, args.out_dir / f"{split}.png"):
                log.info("  wrote %s.png", split)
        summ = pareto_summary(result)
        if args.bootstrap > 0:
            attach_bootstrap_cis(summ, arrays, args.stage2_model, args.stage2_dr,
                                 args.stage2_fpr, args.bootstrap, args.bootstrap_seed)
            log.info("  bootstrapped %d resamples for %s", args.bootstrap, split)
        summaries.append(summ)
        log.info("  %s: %s", split, summ["verdict"])

    (args.out_dir / "frontier_summary.json").write_text(json.dumps({
        "stage2_model": args.stage2_model,
        "illustrative_k2_ms": K2_MS_DATASENTINEL,
        "illustrative_ratio": ILLUSTRATIVE_RATIO,
        "proposal_e_target": PROPOSAL_E_TARGET,
        "fpr_targets": FPR_TARGETS,
        "limitations": LIMITATIONS,
        "splits": summaries,
    }, indent=2))
    print(f"\nwrote {len(summaries)} split frontiers -> {args.out_dir}")
    for s in summaries:
        print(f"  {s['split']:<20} {s['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
