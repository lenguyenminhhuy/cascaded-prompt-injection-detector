"""Experiment 3: temperature scaling + threshold search."""

import argparse

from src.utils.io import load_yaml
from src.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation", default="configs/evaluation.yaml")
    args = parser.parse_args()

    model_cfg = load_yaml(args.config)
    eval_cfg = load_yaml(args.evaluation)
    log.info("calibrating %s", model_cfg["name"])
    log.info("fpr targets: %s", eval_cfg["fpr_targets"])

    raise NotImplementedError("calibrate: implement temperature scaling + threshold sweep")


if __name__ == "__main__":
    main()
