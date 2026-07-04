"""Split-builder invariants (PLAN.md Chunk 6)."""

from __future__ import annotations

import random

from src.data.schema import Sample
from src.data.split_builder import SplitBuilder

CFG = {
    "cross_channel_holdout_cells": [["exfiltration", "document_embedded"],
                                    ["tool_misuse", "direct"]],
    "cross_domain_domains": ["browsing", "calendar", "email"],
    "in_dist": {"ratios": {"train": 0.7, "cal": 0.15, "test_in_dist": 0.15},
                "train_injection_ratio": 0.5, "test_in_dist_injection_ratio": 0.15},
    "calibration": {"difficulty_weights": {"ambiguous": 3.0, "hard": 2.0, "easy": 1.0}},
    "exclude_multi_intent_from_cross_channel": True,
}


def _mk(i, label, fam, channel, domain="sql", diff="easy", split="unassigned"):
    return Sample(id=f"x-{i}", rendered_input=f"sample number {i} unique text here",
                  channel=channel, label=label, payload_family=fam, domain=domain,
                  source="synthetic", difficulty=diff, split=split)


def _pool():
    pool = []
    i = 0
    # in-dist positives across several non-holdout cells
    for _ in range(120):
        pool.append(_mk(i, "injected", "instruction_override", "direct")); i += 1
    for _ in range(120):
        pool.append(_mk(i, "injected", "exfiltration", "tool_output")); i += 1
    # holdout-cell positives
    for _ in range(40):
        pool.append(_mk(i, "injected", "exfiltration", "document_embedded")); i += 1
    for _ in range(40):
        pool.append(_mk(i, "injected", "tool_misuse", "direct")); i += 1
    # benign across channels
    for ch in ("direct", "document_embedded", "tool_output"):
        for _ in range(150):
            pool.append(_mk(i, "benign", None, ch)); i += 1
    # cross-domain (pre-stamped)
    for _ in range(20):
        pool.append(_mk(i, "injected", "tool_misuse", "tool_output", domain="email",
                        split="test_cross_domain")); i += 1
    return pool


def test_holdout_cells_not_in_train():
    splits = SplitBuilder(CFG, seed=42).build(_pool())
    hold = {("exfiltration", "document_embedded"), ("tool_misuse", "direct")}
    train_cells = {(s.payload_family, s.channel) for s in splits["train"] if s.label == "injected"}
    assert not (train_cells & hold), "holdout cells leaked into train"
    cc_cells = {(s.payload_family, s.channel) for s in splits["test_cross_channel"]
                if s.label == "injected"}
    assert hold <= cc_cells


def test_cross_domain_isolated():
    splits = SplitBuilder(CFG, seed=42).build(_pool())
    for name, items in splits.items():
        for s in items:
            if s.domain != "sql":
                assert name == "test_cross_domain"


def test_no_cross_split_leakage():
    splits = SplitBuilder(CFG, seed=42).build(_pool())
    seen = {}
    for name, items in splits.items():
        for s in items:
            assert s.id not in seen, f"{s.id} in both {seen.get(s.id)} and {name}"
            seen[s.id] = name


def test_train_roughly_balanced():
    splits = SplitBuilder(CFG, seed=42).build(_pool())
    tr = splits["train"]
    inj = sum(1 for s in tr if s.label == "injected")
    assert abs(inj / len(tr) - 0.5) < 0.05


def test_test_in_dist_imbalanced():
    splits = SplitBuilder(CFG, seed=42).build(_pool())
    t = splits["test_in_dist"]
    inj = sum(1 for s in t if s.label == "injected")
    assert inj / len(t) < 0.30  # realistic deployment imbalance
