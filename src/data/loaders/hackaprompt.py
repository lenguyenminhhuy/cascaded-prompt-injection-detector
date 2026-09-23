"""HackAPrompt -> direct positives (raw payloads).

Reads the locally cached mirror (scripts/data/download_hf.py). The mirror collapses
each playground submission into a single ``text`` field where the attacker's
injection follows the app framing, separated by ``----------`` rules. We extract
the injected segment and classify override vs system-prompt-extraction.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

from src.data.loaders.base import DatasetLoader
from src.data.schema import Sample, make_id
from src.data.taxonomy import difficulty_for, direct_family

_SEP = re.compile(r"-{4,}")


class HackapromptLoader(DatasetLoader):
    name = "hackaprompt"

    def __init__(self, cache: str = "data/raw/hf/hackaprompt.jsonl",
                 max_samples: int | None = None):
        self.cache = Path(cache)
        self.max_samples = max_samples

    @staticmethod
    def _extract_injection(text: str) -> str:
        # Take the last non-empty segment after the separator rules; that is the
        # attacker-controlled portion in the HackAPrompt template.
        parts = [p.strip() for p in _SEP.split(text) if p.strip()]
        if not parts:
            return text.strip()
        cand = parts[-1]
        # Drop trailing app scaffolding lines if the injection is mid-template.
        return cand.strip()

    def load(self) -> Iterator[Sample]:
        if not self.cache.exists():
            return
        seen: set[str] = set()
        n = 0
        with self.cache.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                payload = self._extract_injection(text)
                if not (15 <= len(payload) <= 2000):
                    continue
                key = payload[:200]
                if key in seen:
                    continue
                seen.add(key)
                family = direct_family(payload)
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
                    notes="hackaprompt",
                ).validate()
                n += 1
                if self.max_samples and n >= self.max_samples:
                    return
