"""E8 — two-sided Stage-1 ablation: add an ``unsafe -> DENY`` branch.

The headline cascade (scripts/eval_cascade.py) runs a ONE-SIDED Stage-1: it only
auto-passes confident-benign (p_safe >= theta_safe) and escalates the rest to M2.
Stage-1 never blocks. The architecture diagram, however, draws a two-sided Stage-1
(unsafe -> DENY, safe -> ALLOW, uncertain -> escalate). This script measures what
the DENY branch actually buys, so the diagram and the numbers tell one story.

Routing (two thresholds on the SAME frozen theta_safe):
    p_safe >= theta_safe            -> ALLOW  (auto-pass, as before)
    p_safe <= p_deny (< theta_safe) -> DENY   (Stage-1 blocks, never reaches M2)
    otherwise                       -> ESCALATE to M2 (decides at theta2)

The hard constraint: OVERALL FPR == 1%. A Stage-1 DENY of a benign input is a
false positive that M2 can never undo AND it spends the shared 1% budget, so M2's
theta2 is re-solved on the *remaining* budget. The DENY branch pays off only if the
attacks it blocks are cheaper (fewer benign-FP per attack) than M2 at the margin.

Honesty guards (mirror eval_cascade.py):
  * theta_safe is the SAME cal-frozen value as the one-sided headline (isolates the
    DENY branch's marginal effect; not re-tuned).
  * p_deny is SELECTED ON CAL (max cal DR s.t. cal FPR <= 1%, incl. the "off"
    candidate) and FROZEN, then reported on eval. The eval-optimal DENY frontier is
    also reported, labelled as an in-sample upper bound.
  * DR@fixed-FPR and per-class routing rates are prevalence-invariant, so the
    cost-vs-attack-share crossover varies ONLY escalation (cost), not DR/FPR.

PAYLOAD HYGIENE (CLAUDE.md): reads only label + logit scores; never input text.

    PYTHONPATH=. python scripts/eval_cascade_twosided.py \
        --stage1-dir results_stage1/stage1/qwen2.5-1.5b \
        --stage2-dir results/stage2/mistral-7b-v0.1 \
        --stage1-name qwen2.5-1.5b --stage2-name mistral-7b \
        --out-dir results/analysis/cascade_qwen2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.eval_cascade import (  # noqa: E402
    FPR_TARGET, STAGE1_PARAMS, STAGE2_PARAMS,
    _cascade_point, _grid, _join, _select_theta_safe_on_cal,
)
from scripts.score_stage1_logits import auroc  # noqa: E402
from src.evaluation.metrics import detection_rate_at_fpr  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger("eval_cascade_twosided")

# Attack-share grid for the cost-vs-prevalence crossover (fraction of traffic that
# is attacks). The real benchmark mix is ~0.276.
PREVALENCE_GRID = [0.01, 0.05, 0.10, 0.276, 0.50, 0.70, 0.90, 0.99]


def _twosided_point(s1_ps, s2, y, theta_safe, p_deny, fpr_target):
    """Two-sided cascade metrics at (theta_safe, p_deny). Overall FPR held to target.

    DENY branch blocks escalated rows with p_safe <= p_deny (most-confident-injection).
    Those benign-denials are unfixable FPs and are subtracted from M2's FP budget;
    M2's theta2 is re-solved on the remaining budget over the (now smaller) escalated
    benign pool. p_deny <= 0 reproduces the one-sided cascade exactly.
    """
    is_ben = y == 0
    is_att = y == 1
    n_ben = int(is_ben.sum())
    n_att = int(is_att.sum())
    budget = fpr_target * n_ben

    auto_pass = s1_ps >= theta_safe
    deny = (~auto_pass) & (s1_ps <= p_deny)      # Stage-1 block
    esc = (~auto_pass) & (~deny)                 # escalate to M2

    leaked = int((auto_pass & is_att).sum())     # auto-passed attacks: FN regardless
    s1_att_caught = int((deny & is_att).sum())   # Stage-1 correct blocks
    s1_ben_denied = int((deny & is_ben).sum())   # Stage-1 false positives (unfixable)
    n_eb = int((esc & is_ben).sum())

    remaining = budget - s1_ben_denied           # FP budget left for M2
    if n_eb == 0:
        m2_caught = int((esc & is_att).sum())
        m2_fp = 0
        theta2 = None
        resolvable = True
    else:
        local = min(1.0, max(0.0, remaining) / n_eb)
        d = detection_rate_at_fpr(y[esc], s2[esc], (local,))[str(local)]
        m2_fp = int(round(d["achieved_fpr"] * d["n_benign"]))
        m2_caught = int(round(d["dr"] * d["n_injection"])) if d["n_injection"] else 0
        theta2 = float(d["threshold"]) if math.isfinite(d["threshold"]) else None
        resolvable = bool(d["resolvable"])

    caught = s1_att_caught + m2_caught
    fp = s1_ben_denied + m2_fp
    tpr = caught / n_att if n_att else float("nan")
    fpr = fp / n_ben if n_ben else float("nan")
    esc_rate = float(esc.sum()) / len(y)

    # per-class routing rates (prevalence-invariant; feed the cost crossover)
    esc_frac_benign = float((esc & is_ben).sum()) / n_ben if n_ben else float("nan")
    esc_frac_attack = float((esc & is_att).sum()) / n_att if n_att else float("nan")

    return {
        "theta_safe": float(theta_safe), "p_deny": float(p_deny),
        "escalation_rate": esc_rate, "e2e_tpr": tpr, "e2e_fpr": fpr,
        "leaked_attacks": leaked,
        "stage1_attacks_denied": s1_att_caught, "stage1_benign_denied": s1_ben_denied,
        "m2_caught": m2_caught, "m2_false_positives": m2_fp,
        "caught_total": caught, "false_positives_total": fp,
        "theta2": theta2, "resolvable": resolvable, "n_esc_benign": n_eb,
        "esc_frac_benign": esc_frac_benign, "esc_frac_attack": esc_frac_attack,
        "budget_used_by_stage1": s1_ben_denied, "budget_total": int(round(budget)),
    }


def _deny_grid(s1_ps, auto_pass, n_levels):
    """Candidate p_deny thresholds from the escalated pool's p_safe quantiles.

    p_deny <= 0.0 is included as the 'DENY off' point (one-sided cascade). Higher
    p_deny denies more of the escalated pool (deeper into the uncertain band).
    """
    esc_ps = s1_ps[~auto_pass]
    if esc_ps.size == 0:
        return [0.0]
    qs = np.linspace(0.0, 0.95, n_levels)
    cands = sorted(set(float(np.quantile(esc_ps, q)) for q in qs))
    return [0.0] + [c for c in cands if c > 0.0]


def _select_p_deny_on_cal(s1_ps_c, s2c, y_c, theta_safe, grid, fpr_target):
    """Frozen p_deny: max cal DR among candidates with cal FPR <= target.

    Ties broken toward the CHEAPER point (lower escalation), then less blocking
    (smaller p_deny). The 'off' candidate (p_deny=0) is always in the grid, so a
    DENY branch that never helps selects itself out.
    """
    scored = []
    for pd in grid:
        pt = _twosided_point(s1_ps_c, s2c, y_c, theta_safe, pd, fpr_target)
        if pt["e2e_fpr"] <= fpr_target + 1e-9:
            scored.append((pt["e2e_tpr"], -pt["escalation_rate"], -pd, pd))
    if not scored:
        return 0.0
    scored.sort(reverse=True)
    return scored[0][3]


def _cost_vs_prevalence(one_sided, two_sided, grid):
    """Escalation (=M2 cost) as attack share varies, for one-sided vs two-sided.

    escalation(a) = (1-a) * esc_frac_benign + a * esc_frac_attack. DR/FPR are
    per-class and prevalence-invariant, so only cost moves — this is where the
    DENY branch earns (or fails to earn) its keep in attack-heavy traffic.
    """
    rows = []
    for a in grid:
        e1 = (1 - a) * one_sided["esc_frac_benign"] + a * one_sided["esc_frac_attack"]
        e2 = (1 - a) * two_sided["esc_frac_benign"] + a * two_sided["esc_frac_attack"]
        rows.append({
            "attack_share": a,
            "escalation_one_sided": round(e1, 4),
            "escalation_two_sided": round(e2, 4),
            "m2_calls_saved_pct_points": round((e1 - e2) * 100.0, 2),
        })
    return rows


def evaluate(args) -> dict:
    s1_dir = Path(args.stage1_dir)
    s2_dir = Path(args.stage2_dir)
    cal_labels = ROOT / args.cal_split
    eval_labels = ROOT / args.eval_split

    _, _, s1_ps_c, y_c = _join(s1_dir, "cal_logits", cal_labels)
    _, _, s2_ps_c, _ = _join(s2_dir, "cal_logits", cal_labels)
    _, _, s1_ps_e, y = _join(s1_dir, "eval_logits", eval_labels)
    _, _, s2_ps_e, _ = _join(s2_dir, "eval_logits", eval_labels)

    s2c = 1.0 - s2_ps_c
    s2e = 1.0 - s2_ps_e

    # --- reproduce the frozen one-sided theta_safe (same as headline) ----------
    dr2_cal = detection_rate_at_fpr(y_c, s2c, (FPR_TARGET,))[str(FPR_TARGET)]["dr"]
    grid = _grid(args.theta_safe_lo, args.theta_safe_hi, args.step)
    theta_safe = _select_theta_safe_on_cal(s1_ps_c, s2c, y_c, grid, dr2_cal,
                                            args.keep_frac, FPR_TARGET)
    if theta_safe is None:
        raise SystemExit("no feasible cal theta_safe; run eval_cascade.py first")

    # --- one-sided reference (DENY off) ----------------------------------------
    one_eval = _twosided_point(s1_ps_e, s2e, y, theta_safe, 0.0, FPR_TARGET)

    # --- select + freeze p_deny on CAL, report on EVAL -------------------------
    auto_pass_c = s1_ps_c >= theta_safe
    dgrid = _deny_grid(s1_ps_c, auto_pass_c, args.deny_levels)
    p_deny = _select_p_deny_on_cal(s1_ps_c, s2c, y_c, theta_safe, dgrid, FPR_TARGET)
    two_cal = _twosided_point(s1_ps_c, s2c, y_c, theta_safe, p_deny, FPR_TARGET)
    two_eval = _twosided_point(s1_ps_e, s2e, y, theta_safe, p_deny, FPR_TARGET)

    # --- eval DENY frontier (in-sample upper bound) ----------------------------
    auto_pass_e = s1_ps_e >= theta_safe
    egrid = _deny_grid(s1_ps_e, auto_pass_e, args.deny_levels)
    frontier = [_twosided_point(s1_ps_e, s2e, y, theta_safe, pd, FPR_TARGET) for pd in egrid]
    feasible = [r for r in frontier if r["e2e_fpr"] <= FPR_TARGET + 1e-9]
    eval_opt = max(feasible, key=lambda r: (r["e2e_tpr"], -r["escalation_rate"])) if feasible else None

    # A cal-frozen deny threshold only "helps" if it still HOLDS the FPR budget on
    # eval. Stage-1 denials are fixed by the threshold and cannot be re-tuned to the
    # budget, so an OOD FPR blow-up disqualifies the point (it is no longer a 1%-FPR
    # operating point, so its higher DR is not comparable to the one-sided headline).
    cal_frozen_holds_fpr = bool(two_eval["e2e_fpr"] <= FPR_TARGET + 1e-9)
    cal_frozen_helps = bool(cal_frozen_holds_fpr and p_deny > 0.0
                            and two_eval["e2e_tpr"] > one_eval["e2e_tpr"] + 1e-9)

    # The FAIR verdict: at a strictly-held 1% eval FPR, does the (in-sample optimal)
    # deny branch beat one-sided? This is the honest upper bound on what DENY buys.
    ub_gain = (eval_opt["e2e_tpr"] - one_eval["e2e_tpr"]) if eval_opt else 0.0
    ub_extra_attacks = (eval_opt["caught_total"] - one_eval["caught_total"]) if eval_opt else 0

    return {
        "stage1_name": args.stage1_name, "stage2_name": args.stage2_name,
        "n": int(len(y)), "n_attacks": int((y == 1).sum()), "n_benign": int((y == 0).sum()),
        "fpr_target": FPR_TARGET, "theta_safe_frozen": theta_safe,
        "stage1_solo_auroc": auroc(y, 1.0 - s1_ps_e),
        "one_sided_eval": one_eval,
        "primary_verdict": {
            "at_strict_1pct_fpr": {
                "deny_dr_upper_bound": eval_opt["e2e_tpr"] if eval_opt else None,
                "one_sided_dr": one_eval["e2e_tpr"],
                "dr_gain_upper_bound": round(ub_gain, 4),
                "extra_attacks_caught_upper_bound": ub_extra_attacks,
            },
            "conclusion": (
                "DENY branch adds ~0 detection at a strictly-held 1% FPR "
                f"(in-sample UB: +{ub_extra_attacks} attacks, +{ub_gain:.4f} DR). "
                "One-sided routing is confirmed as the right operating design."
                if ub_extra_attacks <= max(5, int(0.001 * (y == 1).sum()))
                else f"DENY branch adds +{ub_extra_attacks} attacks at strict 1% FPR (in-sample UB)."),
        },
        "two_sided_cal_frozen": {
            "p_deny": p_deny, "on_cal": two_cal, "on_eval": two_eval,
            "eval_fpr_holds_1pct": cal_frozen_holds_fpr,
            "deny_branch_helps_detection": cal_frozen_helps,
            "verdict": (
                "DENY branch IMPROVES detection AND holds 1% FPR on eval" if cal_frozen_helps else
                "DENY branch selected OFF (p_deny=0): does not beat one-sided at 1% FPR"
                if p_deny == 0.0 else
                f"DENY branch DOES NOT hold the FPR budget OOD: cal threshold p_deny={p_deny:.3g} "
                f"gives eval FPR={two_eval['e2e_fpr']:.4f} (target {FPR_TARGET:.0%}), driven by "
                f"{two_eval['stage1_benign_denied']} Stage-1 benign-denials it cannot re-tune. "
                f"Its DR={two_eval['e2e_tpr']:.4f} is at that inflated FPR, not comparable to the "
                "one-sided headline. Same OOD threshold-transfer failure as the calibration bound."),
        },
        "eval_optimal_deny_upper_bound": (
            {**eval_opt, "note": "IN-SAMPLE upper bound: p_deny argmax'd on eval; optimistic, "
                                 "strictly holds eval FPR<=1%"}
            if eval_opt else {"feasible": False}),
        "cost_vs_prevalence": {
            "operating_point": "two-sided cal-frozen point",
            "two_sided_eval_fpr": two_eval["e2e_fpr"],
            "caveat": ("cost savings below are at the two-sided point's ACTUAL eval FPR "
                       f"({two_eval['e2e_fpr']:.4f}), not 1%. At a strictly-held 1% FPR the deny "
                       "branch is ~off, so its cost saving collapses to ~the one-sided value. "
                       "The deny branch trades false-positive budget for M2 cost."),
            "rows": _cost_vs_prevalence(one_eval, two_eval, PREVALENCE_GRID),
        },
        "provenance": {
            "stage1_dir": str(s1_dir), "stage2_dir": str(s2_dir),
            "routing": "two-sided: auto-pass (p_safe>=theta_safe) | DENY (p_safe<=p_deny) | escalate",
            "theta_safe_source": "cal-frozen, identical to one-sided headline",
        },
    }


def _print(r: dict) -> None:
    one = r["one_sided_eval"]
    ts = r["two_sided_cal_frozen"]
    tw = ts["on_eval"]
    print(f"\n=== TWO-SIDED ABLATION  {r['stage1_name']} -> {r['stage2_name']} ===")
    print(f"eval n={r['n']}  attacks={r['n_attacks']}  benign={r['n_benign']}  "
          f"FPR target={r['fpr_target']:.0%}  theta_safe(frozen)={r['theta_safe_frozen']}")
    print(f"Stage-1 solo AUROC={r['stage1_solo_auroc']:.4f}\n")
    hdr = f"{'':22}{'DR@1%FPR':>10}{'FPR':>9}{'escal.':>9}{'S1 catch':>10}{'S1 benFP':>10}{'M2 catch':>10}"
    print(hdr)
    print(f"{'one-sided (DENY off)':22}{one['e2e_tpr']:>10.4f}{one['e2e_fpr']:>9.4f}"
          f"{one['escalation_rate']:>9.4f}{one['stage1_attacks_denied']:>10}"
          f"{one['stage1_benign_denied']:>10}{one['m2_caught']:>10}")
    print(f"{'two-sided (cal-frozen)':22}{tw['e2e_tpr']:>10.4f}{tw['e2e_fpr']:>9.4f}"
          f"{tw['escalation_rate']:>9.4f}{tw['stage1_attacks_denied']:>10}"
          f"{tw['stage1_benign_denied']:>10}{tw['m2_caught']:>10}")
    print(f"\n  cal-frozen p_deny={ts['p_deny']:.6g}  holds 1% FPR on eval? "
          f"{ts['eval_fpr_holds_1pct']}")
    print(f"  -> {ts['verdict']}")
    eo = r["eval_optimal_deny_upper_bound"]
    if eo.get("feasible") is not False:
        print(f"\n  eval-optimal DENY (in-sample UB, strict 1% FPR): DR={eo['e2e_tpr']:.4f} @ "
              f"p_deny={eo['p_deny']:.6g}, escal={eo['escalation_rate']:.4f}, "
              f"S1 catch={eo['stage1_attacks_denied']}, S1 benFP={eo['stage1_benign_denied']}")
    pv = r["primary_verdict"]["at_strict_1pct_fpr"]
    print(f"\n  *** PRIMARY (strict 1% FPR): DENY upper-bound DR={pv['deny_dr_upper_bound']:.4f} vs "
          f"one-sided {pv['one_sided_dr']:.4f}  (+{pv['extra_attacks_caught_upper_bound']} attacks) ***")
    print(f"  {r['primary_verdict']['conclusion']}")
    cvp = r["cost_vs_prevalence"]
    print(f"\ncost vs attack prevalence (escalation=M2 calls) @ two-sided eval FPR="
          f"{cvp['two_sided_eval_fpr']:.4f} (NOT 1%):")
    print(f"  {'attack%':>8}{'1-sided':>10}{'2-sided':>10}{'saved pp':>10}")
    for row in cvp["rows"]:
        print(f"  {row['attack_share']*100:>7.0f}%{row['escalation_one_sided']:>10.4f}"
              f"{row['escalation_two_sided']:>10.4f}{row['m2_calls_saved_pct_points']:>10.2f}")


def parse_args():
    p = argparse.ArgumentParser(description="E8 two-sided Stage-1 (DENY branch) ablation")
    p.add_argument("--stage1-dir", required=True)
    p.add_argument("--stage2-dir", required=True)
    p.add_argument("--stage1-name", default="stage1")
    p.add_argument("--stage2-name", default="stage2")
    p.add_argument("--cal-split", default="data/train_proposal/cal.jsonl")
    p.add_argument("--eval-split", default="data/eval_proposal/eval.jsonl")
    p.add_argument("--theta-safe-lo", type=float, default=0.0)
    p.add_argument("--theta-safe-hi", type=float, default=1.0)
    p.add_argument("--step", type=float, default=0.005)
    p.add_argument("--keep-frac", type=float, default=0.99)
    p.add_argument("--deny-levels", type=int, default=40,
                   help="number of p_deny quantile candidates from the escalated pool")
    p.add_argument("--stage2-params", type=float, default=STAGE2_PARAMS)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/analysis/cascade")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    res = evaluate(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "twosided_summary.json").write_text(json.dumps(res, indent=2))
    _print(res)
    print(f"\nwrote -> {args.out_dir / 'twosided_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
