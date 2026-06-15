import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from src.data.loaders.base import DatasetLoader
from src.data.schema import Sample

_DATA_FILES = [
    "test_cases_dh_base.json",
    "test_cases_dh_enhanced.json",
    "test_cases_ds_base.json",
    "test_cases_ds_enhanced.json",
]


class InjecAgentLoader(DatasetLoader):
    name = "injecagent"

    def __init__(
        self,
        data_dir: str = "../datasets/InjecAgent/data",
        split: Literal["train", "eval"] = "train",
        seed: int = 42,
        max_samples: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.seed = seed
        self.max_samples = max_samples

    def _load_all_records(self) -> list[dict]:
        records: list[dict] = []
        for fname in _DATA_FILES:
            fpath = self.data_dir / fname
            if not fpath.exists():
                continue
            with fpath.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                records.extend(data)
        return records

    def _split_records(self, records: list[dict]) -> list[dict]:
        rng = random.Random(self.seed)
        indices = list(range(len(records)))
        rng.shuffle(indices)
        cutoff = int(len(indices) * 0.8)
        train_indices = set(indices[:cutoff])
        if self.split == "train":
            return [records[i] for i in range(len(records)) if i in train_indices]
        else:
            return [records[i] for i in range(len(records)) if i not in train_indices]

    def load(self) -> Iterator[Sample]:
        records = self._load_all_records()
        subset = self._split_records(records)
        count = 0
        for row in subset:
            text = (row.get("Tool Response") or "").strip()
            if not text:
                # Fallback: use attacker instruction embedded in context
                text = (row.get("Attacker Instruction") or "").strip()
            if not text:
                continue
            yield Sample(
                input=text,
                label="injection",
                channel="tool_output",
                source=self.name,
                metadata={
                    "attack_type": row.get("Attack Type", ""),
                    "user_instruction": row.get("User Instruction", ""),
                },
            )
            count += 1
            if self.max_samples and count >= self.max_samples:
                break
