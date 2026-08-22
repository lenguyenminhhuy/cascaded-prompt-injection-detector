"""Tail diagnostics — the discrimination wall, calibration-invariant. NO GPU.

The cascade cost floor is a RANK quantity: no monotone recalibration (temperature,
isotonic) can move it. For zero attack-leak the safe threshold must sit above the
highest-scoring attack; for zero benign-FP the block threshold must sit below the
lowest-scoring benign. Everything in the overlap band must escalate. This script
quantifies that band per split — the concrete evidence separating "calibration-
fixable" shifts (empty tails, separable) from "needs a better Stage-1" (non-empty
confidently-safe-attack tail).

PAYLOAD HYGIENE: reads only labels + p_safe. Emits counts / ratios only.

    PYTHONPATH=. python scripts/tail_diagnostics.py --logits-dir scratchpad/e2score2
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
from src.utils.logging import get_logger  # noqa: E402

log = get_logger("tail_diagnostics")

DEFAULT_SPLITS = ["test_in_dist", "test_cross_channel", "test_cross_domain"]
SPLIT_MAP = {
    "val": ROOT / "data/train_proposal/val.jsonl",
    "cal": ROOT / "data/train_proposal/cal.jsonl",
    "test_in_dist": ROOT / "data/splits/test_in_dist.jsonl",
    "test_cross_channel": ROOT / "data/splits/test_cross_channel.jsonl",
    "test_cross_domain": ROOT / "data/splits/test_cross_domain.jsonl",
}
SAFE_TAIL = 0.90    # attacks at/above this "look confidently safe"
UNSAFE_TAIL = 0.05  # benign at/below this "look confidently unsafe"


def diagnose(labels: np.ndarray, p_safe: np.ndarray) -> dict:
    is_attack = labels == 1
    is_benign = labels == 0
    n = int(len(labels))
    n_att = int(is_attack.sum())
    n_ben = int(is_benign.sum())
    atk_ps = p_safe[is_attack]
    ben_ps = p_safe[is_benign]

    max_atk = float(atk_ps.max()) if n_att else float("nan")   # forces theta_safe up
    min_ben = float(ben_ps.min()) if n_ben else float("nan")   # forces theta_unsafe down

    # Calibration-invariant cost floor at zero-leak AND zero benign-FP:
    #   auto-pass region  p_safe >  max attack score  (only benign qualify)
    #   auto-block region p_safe <  min benign score  (only attacks qualify)
    #   everything else MUST escalate.
    n_autopass = int((p_safe > max_atk).sum()) if n_att else n
    n_autoblock = int((p_safe < min_ben).sum()) if n_ben else n
    esc_floor = (n - n_autopass - n_autoblock) / n if n else float("nan")

    return {
        "n": n, "n_attacks": n_att, "n_benign": n_ben,
        "confidently_safe_attacks": {
            "count": int((atk_ps >= SAFE_TAIL).sum()) if n_att else 0,
            "fraction": float((atk_ps >= SAFE_TAIL).mean()) if n_att else 0.0,
            "max_attack_p_safe": max_atk,
        },
        "confidently_unsafe_benign": {
            "count": int((ben_ps <= UNSAFE_TAIL).sum()) if n_ben else 0,
            "fraction": float((ben_ps <= UNSAFE_TAIL).mean()) if n_ben else 0.0,
            "min_benign_p_safe": min_ben,
        },
        "rank_cost_floor": {
            "auto_pass_benign": n_autopass,
            "auto_block_attacks": n_autoblock,
            "escalation_floor": esc_floor,
            "separable": bool(min_ben > max_atk) if (n_att and n_ben) else None,
            "note": ("min escalation to leak 0 attacks AND wrongly block 0 benign; "
                     "invariant to any monotone recalibration."),
        },
    }


def _verdict(d: dict) -> str:
    csa = d["confidently_safe_attacks"]
    fl = d["rank_cost_floor"]
    if csa["count"] == 0 and fl["escalation_floor"] < 0.5:
        return (f"tails light: {fl['escalation_floor']*100:.0f}% forced-escalation floor — "
                f"calibration/threshold placement can help here.")
    return (f"discrimination wall: {csa['count']} attack(s) at p_safe>={SAFE_TAIL:g} "
            f"(max {csa['max_attack_p_safe']:.3f}); escalation floor "
            f"{fl['escalation_floor']*100:.0f}% — no monotone recalibration crosses this.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-1 tail / discrimination diagnostics")
    p.add_argument("--logits-dir", type=Path, default=ROOT / "scratchpad/e2score2")
    p.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    p.add_argument("--out", type=Path,
                   default=ROOT / "results/analysis/calibration/tail_diagnostics.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = {}
    for split in args.splits:
        split_path = SPLIT_MAP.get(split)
        logits_path = args.logits_dir / f"{split}_logits.jsonl"
        if split_path is None or not logits_path.exists() or not split_path.exists():
            log.warning("skip %s (missing %s)", split, logits_path)
            continue
        labels, scores, _s, _c = load_rows(logits_path, split_path)
        labels = np.asarray(labels, dtype=int)
        p_safe = 1.0 - np.asarray(scores, dtype=float)  # load_rows returns 1 - p_safe
        d = diagnose(labels, p_safe)
        d["verdict"] = _verdict(d)
        out[split] = d
        log.info("%s: %s", split, d["verdict"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n{'split':<20} {'safe-atk':>9} {'unsafe-ben':>11} {'esc-floor':>10} {'separable':>10}")
    print("-" * 64)
    for s, d in out.items():
        print(f"{s:<20} {d['confidently_safe_attacks']['count']:>9} "
              f"{d['confidently_unsafe_benign']['count']:>11} "
              f"{d['rank_cost_floor']['escalation_floor']*100:>9.0f}% "
              f"{str(d['rank_cost_floor']['separable']):>10}")
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
