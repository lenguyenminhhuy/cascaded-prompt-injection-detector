"""Experiment 5: detection rate vs escalation rate tradeoff sweep."""

import argparse

from src.utils.io import load_yaml
from src.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", default="configs/evaluation.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.evaluation)
    log.info("sweep range theta_safe=%s theta_unsafe=%s step=%s",
             cfg["threshold_sweep"]["theta_safe_range"],
             cfg["threshold_sweep"]["theta_unsafe_range"],
             cfg["threshold_sweep"]["step"])
    raise NotImplementedError("sweep_thresholds: implement grid sweep + plotting")


if __name__ == "__main__":
    main()
