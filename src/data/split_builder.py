"""Orchestrates data loaders + preprocessing into train/val/cal/eval JSONL splits."""

import logging
import random
from pathlib import Path

from src.data.loaders.base import DatasetLoader
from src.data.preprocessing.augmentation import augment
from src.data.preprocessing.dedup import dedup
from src.data.preprocessing.sampler import balanced_sample
from src.data.schema import Sample
from src.utils.io import write_jsonl

log = logging.getLogger(__name__)


class SplitBuilder:
    def __init__(
        self,
        train_loaders: list[DatasetLoader],
        eval_loaders: list[DatasetLoader],
        out_dir: str = "data/splits",
        seed: int = 42,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        target_injection_ratio: float = 0.5,
    ):
        self.train_loaders = train_loaders
        self.eval_loaders = eval_loaders
        self.out_dir = Path(out_dir)
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.target_injection_ratio = target_injection_ratio

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> dict[str, int]:
        """Run the full pipeline. Returns {split_name: sample_count}."""
        log.info("Loading train pool sources (%d loaders)…", len(self.train_loaders))
        raw_train = self._collect(self.train_loaders)
        log.info("  raw train pool: %d samples", len(raw_train))

        # Preprocessing: dedup → augment injection samples → balance → shuffle
        train_pool = dedup(raw_train)
        log.info("  after dedup: %d samples", len(train_pool))

        train_pool = augment(train_pool)
        log.info("  after augmentation: %d samples", len(train_pool))

        train_pool = balanced_sample(train_pool, ratio=self.target_injection_ratio, seed=self.seed)
        log.info("  after balancing: %d samples", len(train_pool))

        splits = self._make_train_val_cal(train_pool)

        log.info("Loading eval sources (%d loaders)…", len(self.eval_loaders))
        raw_eval = self._collect(self.eval_loaders)
        splits["eval"] = dedup(raw_eval)
        log.info("  eval: %d samples", len(splits["eval"]))

        self.out_dir.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {}
        for name, samples in splits.items():
            path = self.out_dir / f"{name}.jsonl"
            write_jsonl(path, [s.to_dict() for s in samples])
            counts[name] = len(samples)
            log.info("Wrote %s → %d samples", path, counts[name])

        return counts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect(self, loaders: list[DatasetLoader]) -> list[Sample]:
        samples: list[Sample] = []
        for loader in loaders:
            try:
                batch = list(loader.load())
                log.info("  %s: %d samples", loader.name, len(batch))
                samples.extend(batch)
            except Exception as exc:
                log.warning("  %s FAILED: %s — skipping", loader.name, exc)
        return samples

    def _make_train_val_cal(self, pool: list[Sample]) -> dict[str, list[Sample]]:
        rng = random.Random(self.seed)
        shuffled = list(pool)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        return {
            "train": shuffled[:n_train],
            "val": shuffled[n_train : n_train + n_val],
            "cal": shuffled[n_train + n_val :],
        }
