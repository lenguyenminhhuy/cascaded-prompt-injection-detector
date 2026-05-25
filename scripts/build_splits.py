"""Experiment 1: build train/val/cal/eval JSONL splits per Table 2."""

import argparse
from pathlib import Path

from src.utils.io import load_yaml
from src.utils.logging import get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="configs/data/datasets.yaml")
    parser.add_argument("--splits", default="configs/data/splits.yaml")
    parser.add_argument("--out", default="data/splits")
    args = parser.parse_args()

    splits_cfg = load_yaml(args.splits)
    set_seed(splits_cfg.get("seed", 42))

    log.info("datasets: %s", args.datasets)
    log.info("splits cfg: %s", args.splits)
    log.info("output dir: %s", args.out)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    # TODO: load all loaders, apply preprocessing, write JSONL splits.
    raise NotImplementedError("build_splits: wire up loaders + split_builder")


if __name__ == "__main__":
    main()
