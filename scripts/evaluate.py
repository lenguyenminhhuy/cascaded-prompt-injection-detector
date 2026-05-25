"""Experiments 4-6: aggregate metrics, per-channel breakdowns, tables/plots."""

import argparse

from src.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", default="results/predictions")
    parser.add_argument("--metrics-dir", default="results/metrics")
    parser.add_argument("--figures-dir", default="results/figures")
    args = parser.parse_args()
    log.info("aggregating from %s", args.predictions_dir)
    raise NotImplementedError("evaluate: implement metrics aggregation + LaTeX/plot output")


if __name__ == "__main__":
    main()
