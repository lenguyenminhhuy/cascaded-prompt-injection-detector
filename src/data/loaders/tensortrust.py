"""TensorTrust -> direct positives (raw payloads).

Reads the locally cached benchmark JSONL files (scripts/download_hf.py):
  - extraction-robustness: attacks that try to reveal the system prompt
    -> system_prompt_extraction
  - hijacking-robustness: attacks that try to override the bot's behavior
    -> instruction_override (refined per text via taxonomy.direct_family)
The ``attack`` field is the pure attacker user input (direct channel).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from src.data.loaders.base import DatasetLoader
from src.data.schema import Sample, make_id
from src.data.taxonomy import difficulty_for, direct_family

_FILES = [
    ("tensortrust_extraction.jsonl", "system_prompt_extraction"),
    ("tensortrust_hijacking.jsonl", "instruction_override"),
]


class TensorTrustLoader(DatasetLoader):
    name = "tensortrust"

    def __init__(self, cache_dir: str = "data/raw/hf", max_samples: int | None = None):
        self.cache_dir = Path(cache_dir)
        self.max_samples = max_samples

    def load(self) -> Iterator[Sample]:
        seen: set[str] = set()
        n = 0
        for fname, default_family in _FILES:
            p = self.cache_dir / fname
            if not p.exists():
                continue
            with p.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    payload = (row.get("attack") or "").strip()
                    if not (15 <= len(payload) <= 2000):
                        continue
                    key = payload[:200]
                    if key in seen:
                        continue
                    seen.add(key)
                    # Let the text refine the default (some hijacks also extract).
                    family = direct_family(payload)
                    if default_family == "system_prompt_extraction":
                        family = "system_prompt_extraction"
                    yield Sample(
                        id=make_id(self.name, family, n),
                        rendered_input=payload,
                        channel="direct",
                        label="injected",
                        payload_family=family,
                        domain="sql",
                        source=self.name,
                        difficulty=difficulty_for(payload),
                        structural_features=[],
                        notes=f"tensortrust:{fname.split('_')[1].split('.')[0]}",
                    ).validate()
                    n += 1
                    if self.max_samples and n >= self.max_samples:
                        return
