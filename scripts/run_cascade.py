"""Experiment 4: full cascade inference (Stage 1 -> route -> Stage 2)."""

import argparse

from src.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-config", required=True)
    parser.add_argument("--calibration", required=True, help="path to learned T + thresholds")
    parser.add_argument("--eval-split", default="data/splits/eval.jsonl")
    parser.add_argument("--out", default="results/predictions/cascade.jsonl")
    args = parser.parse_args()
    log.info("stage1 config: %s", args.stage1_config)
    log.info("calibration: %s", args.calibration)
    raise NotImplementedError("run_cascade: implement orchestrator")


if __name__ == "__main__":
    main()
