"""InjecAgent -> tool_output positives (raw payloads).

Each test case carries an 'Attacker Instruction' embedded in a tool response.
dh (direct-harm) cases -> tool_misuse; ds (data-stealing) -> exfiltration.
base -> easy, enhanced -> hard. SQL-domain tool wrapping happens in Chunk 3.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from src.data.loaders.base import DatasetLoader
from src.data.schema import Sample, make_id
from src.data.taxonomy import INJECAGENT_SPLIT_FAMILY

_FILES = [
    ("dh", "base", "test_cases_dh_base.json"),
    ("dh", "enhanced", "test_cases_dh_enhanced.json"),
    ("ds", "base", "test_cases_ds_base.json"),
    ("ds", "enhanced", "test_cases_ds_enhanced.json"),
]


class InjecAgentLoader(DatasetLoader):
    name = "injecagent"

    def __init__(self, data_dir: str = "../datasets/InjecAgent/data",
                 max_samples: int | None = None):
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples

    def load(self) -> Iterator[Sample]:
        seen: set[str] = set()
        n = 0
        for split, variant, fname in _FILES:
            p = self.data_dir / fname
            if not p.exists():
                continue
            rows = json.load(p.open(encoding="utf-8"))
            family = INJECAGENT_SPLIT_FAMILY[split]
            diff = "hard" if variant == "enhanced" else "easy"
            for row in rows:
                payload = (row.get("Attacker Instruction") or "").strip()
                if not payload or payload in seen:
                    continue
                seen.add(payload)
                yield Sample(
                    id=make_id(self.name, family, n),
                    rendered_input=payload,
                    channel="tool_output",
                    label="injected",
                    payload_family=family,
                    domain="sql",
                    source=self.name,
                    difficulty=diff,
                    structural_features=[],
                    notes=f"injecagent:{split}:{variant}:{row.get('Attack Type','')}",
                ).validate()
                n += 1
                if self.max_samples and n >= self.max_samples:
                    return
