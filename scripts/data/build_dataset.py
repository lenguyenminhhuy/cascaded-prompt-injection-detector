"""End-to-end cascade-pid v1 dataset build (PLAN.md Chunks 2-6).

Pipeline:
  load positives (5 sources) -> render channel-faithfully -> add hard negatives
  -> synthetic fill of thin/empty cells + ambiguous + cross-domain slice
  -> near-dup dedup -> coverage report -> 5 splits -> contamination check.

Outputs under data/:
  raw_positives.jsonl, pool.jsonl, coverage_report.json,
  splits/{train,cal,test_in_dist,test_cross_channel,test_cross_domain}.jsonl,
  splits/manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

from src.data import negatives as NEG
from src.data import rendering as R
from src.data import synthetic as SYN
from src.data.loaders.agentdojo import AgentDojoLoader
from src.data.loaders.bipia import BipiaLoader
from src.data.loaders.hackaprompt import HackapromptLoader
from src.data.loaders.injecagent import InjecAgentLoader
from src.data.loaders.tensortrust import TensorTrustLoader
from src.data.preprocessing.dedup import dedup
from src.data.preprocessing.dedup import _normalize
from src.data.schema import Sample
from src.data.split_builder import SplitBuilder
from src.utils.io import load_yaml, write_jsonl
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("build")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Synthetic fill: (family, channel, count). Guarantees the cross-channel holdout
# cells exist and the SQL-specific rbac_bypass family is represented everywhere.
FILL_PLAN = [
    ("exfiltration", "document_embedded", 40),
    ("tool_misuse", "direct", 60),
    ("system_prompt_extraction", "tool_output", 60),
    ("system_prompt_extraction", "document_embedded", 30),
    ("rbac_bypass", "direct", 70),
    ("rbac_bypass", "tool_output", 55),
    ("rbac_bypass", "document_embedded", 45),
    ("instruction_override", "tool_output", 45),
]
NEG_PER_CHANNEL = {"direct": 1150, "document_embedded": 850, "tool_output": 850}


def load_positives(injecagent_dir: str, hackaprompt_cap: int) -> list[Sample]:
    loaders = [
        BipiaLoader(local=str(DATA / "raw" / "BIPIA")),
        InjecAgentLoader(data_dir=injecagent_dir),
        AgentDojoLoader(local=str(DATA / "raw" / "agentdojo")),
        HackapromptLoader(cache=str(DATA / "raw" / "hf" / "hackaprompt.jsonl"),
                          max_samples=hackaprompt_cap),
        TensorTrustLoader(cache_dir=str(DATA / "raw" / "hf")),
    ]
    out: list[Sample] = []
    for L in loaders:
        batch = list(L.load())
        log.info("  loaded %-12s %d", L.name, len(batch))
        out.extend(batch)
    return out


def render_positives(positives: list[Sample], seed: int) -> list[Sample]:
    """Replace bare payloads with channel-faithful rendered_input (Chunk 3)."""
    rng = random.Random(seed + 1)
    rendered = []
    dropped = 0
    for s in positives:
        payload = s.rendered_input
        text, sf = R.render(s.channel, payload, rng)
        if not R.payload_preserved(payload, text):
            dropped += 1
            continue
        s.rendered_input = text
        s.structural_features = sf
        rendered.append(s.validate())
    if dropped:
        log.warning("  dropped %d positives where payload not preserved", dropped)
    return rendered


def coverage(samples: list[Sample]) -> dict:
    cells = Counter((s.payload_family or "benign", s.channel) for s in samples)
    return {
        "total": len(samples),
        "by_label": dict(Counter(s.label for s in samples)),
        "by_channel": dict(Counter(s.channel for s in samples)),
        "by_family": dict(Counter(s.payload_family for s in samples)),
        "by_difficulty": dict(Counter(s.difficulty for s in samples)),
        "by_source": dict(Counter(s.source for s in samples)),
        "by_domain": dict(Counter(s.domain for s in samples)),
        "cells": {f"{fam}|{ch}": n for (fam, ch), n in sorted(cells.items())},
    }


def contamination_check(splits: dict[str, list[Sample]]) -> dict:
    """No normalized rendered_input may appear in both train and any test split."""
    train_norm = {_normalize(s.rendered_input) for s in splits.get("train", [])}
    report = {}
    for name in ("test_in_dist", "test_cross_channel", "test_cross_domain"):
        leaks = sum(1 for s in splits.get(name, [])
                    if _normalize(s.rendered_input) in train_norm)
        report[name] = leaks
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-cfg", default=str(ROOT / "configs/data/splits.yaml"))
    ap.add_argument("--injecagent-dir", default=str(ROOT.parent / "datasets/InjecAgent/data"))
    ap.add_argument("--hackaprompt-cap", type=int, default=600)
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args()

    cfg = load_yaml(args.splits_cfg)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    out = Path(args.out)
    (out / "splits").mkdir(parents=True, exist_ok=True)

    log.info("[1/7] loading positives")
    positives = load_positives(args.injecagent_dir, args.hackaprompt_cap)
    write_jsonl(out / "raw_positives.jsonl", [s.to_dict() for s in positives])

    log.info("[2/7] rendering positives channel-faithfully")
    positives = render_positives(positives, seed)

    log.info("[3/7] hard benign negatives")
    rng = random.Random(seed + 2)
    neg = NEG.build_negatives(rng, NEG_PER_CHANNEL)

    log.info("[4/7] synthetic fill + ambiguous + cross-domain")
    rng_s = random.Random(seed + 3)
    syn = SYN.fill_cells(rng_s, FILL_PLAN)
    syn += SYN.ambiguous_cases(rng_s, repeats=14)
    cross = SYN.cross_domain_slice(rng_s, repeats=12)

    pool = positives + neg + syn + cross
    log.info("  combined pool: %d", len(pool))

    log.info("[5/7] near-dup dedup")
    before = len(pool)
    pool = dedup(pool)
    log.info("  %d -> %d (removed %d)", before, len(pool), before - len(pool))
    write_jsonl(out / "pool.jsonl", [s.to_dict() for s in pool])

    cov = coverage(pool)
    (out / "coverage_report.json").write_text(json.dumps(cov, indent=2))
    log.info("  coverage: %s", cov["by_label"])

    log.info("[6/7] building splits")
    splits = SplitBuilder(cfg, seed=seed).build(pool)
    manifest = {"seed": seed, "splits": {}}
    for name, items in splits.items():
        write_jsonl(out / "splits" / f"{name}.jsonl", [s.to_dict() for s in items])
        manifest["splits"][name] = coverage(items)
        log.info("  %-20s %5d  %s", name, len(items),
                 dict(Counter(s.label for s in items)))

    log.info("[7/7] contamination check")
    contam = contamination_check(splits)
    manifest["contamination"] = contam
    log.info("  cross-split leaks: %s", contam)

    (out / "splits" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("DONE")


if __name__ == "__main__":
    main()
