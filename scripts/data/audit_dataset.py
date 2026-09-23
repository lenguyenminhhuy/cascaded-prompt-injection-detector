"""Quality audit + balance/base-rate report (PLAN.md Chunk 7.1-7.3).

Checks, all written to data/audit/:
  - per-channel / per-split balance and base rates           -> balance_report.json
  - structural-feature distribution by label (anti-cheating) -> structural_audit.json
  - all-pairs cross-split leakage on normalized text         -> leakage_report.json
  - a per-channel manual-audit sample                        -> manual_audit_sample.jsonl
  - schema re-validation of every example                    -> (raises on failure)

Exit code is non-zero if any hard invariant fails (leakage > 0, schema invalid,
or a structural feature that appears in only one class).
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.data.preprocessing.dedup import _normalize
from src.data.schema import Sample

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "splits"
AUDIT = ROOT / "data" / "audit"
SPLIT_NAMES = ["train", "cal", "test_in_dist", "test_cross_channel", "test_cross_domain"]


def _load(name: str) -> list[Sample]:
    path = SPLITS / f"{name}.jsonl"
    return [Sample.from_dict(json.loads(l)) for l in path.open(encoding="utf-8")]


def balance_report(splits: dict[str, list[Sample]]) -> dict:
    rep = {}
    for name, items in splits.items():
        n = len(items)
        inj = sum(1 for s in items if s.label == "injected")
        per_ch = defaultdict(lambda: {"injected": 0, "benign": 0})
        for s in items:
            per_ch[s.channel][s.label] += 1
        rep[name] = {
            "n": n,
            "injected": inj,
            "benign": n - inj,
            "base_rate_injected": round(inj / n, 4) if n else 0,
            "per_channel": {k: dict(v) for k, v in per_ch.items()},
            "per_family": dict(Counter(s.payload_family for s in items if s.label == "injected")),
        }
    return rep


def structural_audit(splits: dict[str, list[Sample]]) -> tuple[dict, list[str]]:
    """A structural feature present in only ONE label class lets the model cheat."""
    by_feat = defaultdict(lambda: {"injected": 0, "benign": 0})
    everything = [s for items in splits.values() for s in items]
    for s in everything:
        for f in s.structural_features:
            by_feat[f][s.label] += 1
    problems = []
    for feat, c in by_feat.items():
        # zero-width/white_on_white/tiny_font are legitimately injection-only
        # (hiding text IS the signal); flag only the benign-plausible ones.
        benign_plausible = {"json_field", "db_row", "metadata_description",
                            "html_comment", "ddl_comment", "code_block",
                            "quoted_text", "chunk_position"}
        if feat in benign_plausible and (c["injected"] == 0 or c["benign"] == 0):
            problems.append(f"{feat}: appears in only one class {dict(c)}")
    return {k: dict(v) for k, v in by_feat.items()}, problems


def leakage_report(splits: dict[str, list[Sample]]) -> tuple[dict, int]:
    norm = {name: {_normalize(s.rendered_input) for s in items}
            for name, items in splits.items()}
    rep, total = {}, 0
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = len(norm[a] & norm[b])
            rep[f"{a}__vs__{b}"] = overlap
            total += overlap
    return rep, total


def manual_sample(splits: dict[str, list[Sample]], per_channel: int = 6) -> list[dict]:
    rng = random.Random(7)
    out = []
    for name, items in splits.items():
        by_ch = defaultdict(list)
        for s in items:
            by_ch[s.channel].append(s)
        for ch, group in by_ch.items():
            rng.shuffle(group)
            for s in group[:per_channel]:
                d = s.to_dict()
                d["_audit_split"] = name
                out.append(d)
    return out


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    splits = {n: _load(n) for n in SPLIT_NAMES}

    # 0) schema re-validation
    invalid = 0
    for items in splits.values():
        for s in items:
            try:
                s.validate()
            except Exception as e:  # noqa: BLE001
                invalid += 1
                # cap the message: a validation error can embed raw row text,
                # which must not be echoed into terminals/agent sessions
                print(f"INVALID: id={s.id!r} {type(e).__name__}: {str(e)[:120]}")
    print(f"schema re-validation: {invalid} invalid")

    bal = balance_report(splits)
    (AUDIT / "balance_report.json").write_text(json.dumps(bal, indent=2))

    struct, struct_problems = structural_audit(splits)
    (AUDIT / "structural_audit.json").write_text(
        json.dumps({"by_feature": struct, "problems": struct_problems}, indent=2))

    leak, leak_total = leakage_report(splits)
    (AUDIT / "leakage_report.json").write_text(json.dumps(leak, indent=2))

    sample = manual_sample(splits)
    with (AUDIT / "manual_audit_sample.jsonl").open("w", encoding="utf-8") as f:
        for d in sample:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"leakage total: {leak_total}")
    print(f"structural problems: {struct_problems or 'none'}")
    print(f"manual sample: {len(sample)} rows -> data/audit/manual_audit_sample.jsonl")
    for n in SPLIT_NAMES:
        print(f"  {n:20s} n={bal[n]['n']:5d}  inj_rate={bal[n]['base_rate_injected']}")

    ok = invalid == 0 and leak_total == 0 and not struct_problems
    print("AUDIT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
