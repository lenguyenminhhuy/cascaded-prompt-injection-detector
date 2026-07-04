"""BIPIA -> document_embedded positives (raw payloads).

BIPIA stores attack strings grouped by category in
``benchmark/{text,code}_attack_{train,test}.json`` as {category: [strings]}.
We take the unique attack strings and map each category to a payload_family.
Channel-faithful document wrapping happens in Chunk 3.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from src.data.loaders.base import DatasetLoader
from src.data.schema import Sample, make_id
from src.data.taxonomy import (
    BIPIA_CODE_EXFIL,
    BIPIA_CODE_FAMILY_DEFAULT,
    BIPIA_HARD_TEXT,
    BIPIA_TEXT_FAMILY,
    difficulty_for,
)


class BipiaLoader(DatasetLoader):
    name = "bipia"

    def __init__(self, local: str = "data/raw/BIPIA", max_samples: int | None = None):
        self.root = Path(local) / "benchmark"
        self.max_samples = max_samples

    def _read(self, fname: str) -> dict[str, list[str]]:
        p = self.root / fname
        if not p.exists():
            return {}
        return json.load(p.open(encoding="utf-8"))

    def load(self) -> Iterator[Sample]:
        seen: set[str] = set()
        n = 0
        groups = [
            ("text", self._read("text_attack_train.json")),
            ("text", self._read("text_attack_test.json")),
            ("code", self._read("code_attack_train.json")),
            ("code", self._read("code_attack_test.json")),
        ]
        for kind, data in groups:
            for category, payloads in data.items():
                for payload in payloads:
                    payload = (payload or "").strip()
                    if not payload or payload in seen:
                        continue
                    seen.add(payload)
                    if kind == "text":
                        family = BIPIA_TEXT_FAMILY.get(category, "instruction_override")
                        hard = category in BIPIA_HARD_TEXT
                    else:
                        family = ("exfiltration" if category in BIPIA_CODE_EXFIL
                                  else BIPIA_CODE_FAMILY_DEFAULT)
                        hard = True  # code attacks are inherently obfuscated/technical
                    diff = "hard" if hard else difficulty_for(payload)
                    yield Sample(
                        id=make_id(self.name, family, n),
                        rendered_input=payload,
                        channel="document_embedded",
                        label="injected",
                        payload_family=family,
                        domain="sql",
                        source=self.name,
                        difficulty=diff,
                        structural_features=[],
                        notes=f"bipia:{kind}:{category}",
                    ).validate()
                    n += 1
                    if self.max_samples and n >= self.max_samples:
                        return
