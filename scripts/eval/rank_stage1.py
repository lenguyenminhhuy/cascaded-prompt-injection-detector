"""Rank E2 Stage-1 candidates across all splits — NO GPU, payload-safe.

Scans a results dir for per-candidate logit dumps (<split>_logits.jsonl),
joins each to its split's labels by row order, and prints one comparison table
per split. Ranking key defaults to test_cross_domain AUROC (ECE tie-break) — the
split that defeats the source-style shortcut (all 'authored', both labels), so
it, not the saturated val AUROC, is what actually discriminates candidates.
AUROC is used over accuracy@0.5 because the shifted splits are badly
miscalibrated (accuracy can look poor while ranking/AUROC is strong).

    PYTHONPATH=. python scripts/eval/rank_stage1.py --results-dir results_stage1/stage1

Splits are auto-mapped by logit-file stem; override paths with --split-map if
your layout differs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.score_stage1_logits import compute_metrics, load_rows  # noqa: E402

# logit-file stem -> split jsonl providing the labels
DEFAULT_SPLIT_MAP = {
    "val": ROOT / "data/train_proposal/val.jsonl",
    "cal": ROOT / "data/train_proposal/cal.jsonl",
    "test_in_dist": ROOT / "data/splits/test_in_dist.jsonl",
    "test_cross_channel": ROOT / "data/splits/test_cross_channel.jsonl",
    "test_cross_domain": ROOT / "data/splits/test_cross_domain.jsonl",
}
RANK_SPLIT = "test_cross_domain"  # the shortcut-proof discriminator


def _fmt(x) -> str:
    return "  nan " if x != x else f"{x:.4f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank Stage-1 candidates (no GPU)")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results_stage1/stage1")
    p.add_argument("--rank-split", default=RANK_SPLIT,
                   help=f"split whose AUROC (ECE tie-break) ranks candidates "
                        f"(default {RANK_SPLIT})")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    split_map = {k: v for k, v in DEFAULT_SPLIT_MAP.items()}

    candidates = sorted(d.name for d in args.results_dir.iterdir() if d.is_dir())
    if not candidates:
        raise SystemExit(f"no candidate dirs under {args.results_dir}")

    # metrics[split][candidate] = dict
    metrics: dict[str, dict[str, dict]] = {s: {} for s in split_map}
    for cand in candidates:
        for split_key, split_path in split_map.items():
            logits = args.results_dir / cand / f"{split_key}_logits.jsonl"
            if not logits.exists() or not split_path.exists():
                continue
            labels, scores, sources, channels = load_rows(logits, split_path)
            metrics[split_key][cand] = compute_metrics(labels, scores, sources, channels)

    hdr = f"{'candidate':<22} {'AUROC':>7} {'acc':>7} {'DR@1%':>7} {'ECE':>7} {'n':>6}"
    for split_key in split_map:
        rows = metrics[split_key]
        if not rows:
            continue
        star = "  <-- ranks candidates" if split_key == args.rank_split else ""
        print(f"\n### {split_key}{star}\n{hdr}\n{'-'*len(hdr)}")
        for cand in candidates:
            m = rows.get(cand)
            if m is None:
                print(f"{cand:<22} {'(missing logits)':>39}")
                continue
            dr1 = m["detection_rate_at_fpr"].get("0.01", {}).get("dr", float("nan"))
            print(f"{cand:<22} {_fmt(m['auroc'])} {_fmt(m['accuracy'])} "
                  f"{_fmt(dr1)} {_fmt(m['ece'])} {m['n']:>6}")

    rank = metrics.get(args.rank_split, {})
    if rank:
        # Rank on shifted-split AUROC (robust under miscalibration), ECE as
        # tie-break (lower = better). accuracy@0.5 is misleading here: e.g.
        # cross_domain acc can sit at 0.59 while AUROC is 0.93 (all p_safe<0.5).
        def _key(c: str) -> tuple:
            au = rank[c]["auroc"]
            ec = rank[c]["ece"]
            au = -1.0 if au != au else au          # NaN AUROC -> worst
            ec = float("inf") if ec != ec else ec  # NaN ECE   -> worst tiebreak
            return (au, -ec)
        winner = max(rank, key=_key)
        print(f"\nWINNER by {args.rank_split} AUROC (ECE tie-break): {winner} "
              f"(AUROC={_fmt(rank[winner]['auroc'])}, "
              f"ECE={_fmt(rank[winner]['ece'])}, "
              f"acc={_fmt(rank[winner]['accuracy'])})")
    else:
        print(f"\n(no {args.rank_split} logits yet — winner undecided; "
              f"score the test splits first)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
