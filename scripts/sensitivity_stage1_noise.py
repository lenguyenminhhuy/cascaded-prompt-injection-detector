#!/usr/bin/env python3
"""Stage-1 score-perturbation sensitivity of the cascade.

Motivation. Table 6's bf16 Stage-1 block reuses the escalation rate `e` measured
under NF4, and no detection was scored in bf16. Both DR and `e` are functions of
the Stage-1 confidence scores, so a precision change perturbs them jointly. This
script quantifies how much Stage-1 score drift the reported result can absorb
before the pre-registered criteria fail -- without needing a GPU.

Method. Perturb the Stage-1 logit gap z = logp_benign - logp_injection with
Gaussian noise of scale sigma on BOTH the calibration and evaluation splits
(so theta_safe re-selection sees the same perturbed model, as it would in a real
re-run), recompute p_safe = sigmoid(z), re-select theta_safe on cal by the frozen
procedure, and evaluate on eval. Stage-2 (M2) scores are untouched: it stays NF4
in both Table 6 blocks.

Reports DR@1%FPR, escalation rate e, and leaked attacks vs sigma, with the
per-sigma spread over seeds.
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.eval_cascade import (  # noqa: E402
    _cascade_point, _select_theta_safe_on_cal, _grid, _join, FPR_TARGET,
)
from src.evaluation.metrics import detection_rate_at_fpr  # noqa: E402


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def run(args):
    s1 = Path(args.stage1_dir); s2 = Path(args.stage2_dir)
    cal_lbl = ROOT / args.cal_split; ev_lbl = ROOT / args.eval_split

    lb_c, li_c, _, y_c = _join(s1, "cal_logits", cal_lbl)
    lb_e, li_e, _, y_e = _join(s1, "eval_logits", ev_lbl)
    _, _, s2c, _ = _join(s2, "cal_logits", cal_lbl)
    _, _, s2e, _ = _join(s2, "eval_logits", ev_lbl)

    # Stage-2 is scored as p_injection = 1 - p_safe by eval_cascade convention.
    s2c = 1.0 - s2c
    s2e = 1.0 - s2e

    z_c = lb_c - li_c
    z_e = lb_e - li_e

    grid = _grid(args.theta_safe_lo, args.theta_safe_hi, args.step)
    dr2_cal = detection_rate_at_fpr(y_c, s2c, (FPR_TARGET,))[str(FPR_TARGET)]["dr"]
    m2_alone = detection_rate_at_fpr(y_e, s2e, (FPR_TARGET,))[str(FPR_TARGET)]["dr"]

    print(f"stage1={s1.name}  n_cal={len(y_c)}  n_eval={len(y_e)}")
    print(f"M2-on-every-input DR@{FPR_TARGET:.0%}FPR = {m2_alone:.4f}   "
          f"(cal DR used for theta selection = {dr2_cal:.4f})\n")

    rows = []
    for sigma in args.sigmas:
        n_seeds = 1 if sigma == 0 else args.n_seeds
        drs, es, leaks, thetas = [], [], [], []
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 + seed)
            if sigma == 0:
                zc, ze = z_c, z_e
            else:
                zc = z_c + rng.normal(0.0, sigma, size=z_c.shape)
                ze = z_e + rng.normal(0.0, sigma, size=z_e.shape)
            ps_c, ps_e = _sigmoid(zc), _sigmoid(ze)
            th = _select_theta_safe_on_cal(ps_c, s2c, y_c, grid, dr2_cal,
                                           args.keep_frac, FPR_TARGET)
            if th is None:
                continue
            pt = _cascade_point(ps_e, s2e, y_e, th, FPR_TARGET)
            drs.append(pt["e2e_tpr"]); es.append(pt["escalation_rate"])
            leaks.append(pt["leaked_attacks"]); thetas.append(th)
        if not drs:
            print(f"sigma={sigma:<6} NO FEASIBLE theta_safe on cal")
            continue
        rows.append({
            "sigma": sigma, "n_ok": len(drs),
            "theta_safe_mean": float(np.mean(thetas)),
            "dr_mean": float(np.mean(drs)), "dr_sd": float(np.std(drs)),
            "e_mean": float(np.mean(es)), "e_sd": float(np.std(es)),
            "leaked_mean": float(np.mean(leaks)),
            "delta_dr_vs_m2": float(np.mean(drs) - m2_alone),
        })

    hdr = (f"{'sigma':>7} {'seeds':>5} {'theta':>7} {'DR':>16} {'e':>16} "
           f"{'leaked':>8} {'dDR vs M2':>10} {'2pt?':>5}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        ok = "ok" if abs(r["delta_dr_vs_m2"]) <= 0.02 else "FAIL"
        print(f"{r['sigma']:>7.3g} {r['n_ok']:>5} {r['theta_safe_mean']:>7.3f} "
              f"{r['dr_mean']:>7.4f}+-{r['dr_sd']:<7.4f} "
              f"{r['e_mean']:>7.4f}+-{r['e_sd']:<7.4f} "
              f"{r['leaked_mean']:>8.0f} {r['delta_dr_vs_m2']:>+10.4f} {ok:>5}")

    out = {"stage1": s1.name, "m2_alone_dr": m2_alone, "fpr_target": FPR_TARGET,
           "keep_frac": args.keep_frac, "n_seeds": args.n_seeds, "rows": rows}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage1-dir", required=True)
    p.add_argument("--stage2-dir", required=True)
    p.add_argument("--cal-split", default="data/train_proposal/cal.jsonl")
    p.add_argument("--eval-split", default="data/eval_proposal/eval.jsonl")
    p.add_argument("--sigmas", type=float, nargs="+",
                   default=[0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--keep-frac", type=float, default=0.99)
    p.add_argument("--theta-safe-lo", type=float, default=0.0)
    p.add_argument("--theta-safe-hi", type=float, default=1.0)
    p.add_argument("--step", type=float, default=0.005)
    p.add_argument("--out", default=None)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run(parse_args()))
