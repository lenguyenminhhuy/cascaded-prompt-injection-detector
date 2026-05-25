"""Baselines: PromptGuard 2 and LlamaGuard 4 evaluated standalone."""

import argparse

from src.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-split", default="data/splits/eval.jsonl")
    parser.add_argument("--out", default="results/predictions")
    args = parser.parse_args()
    log.info("eval split: %s", args.eval_split)
    raise NotImplementedError("run_baselines: implement PromptGuard 2 + LlamaGuard 4 inference")


if __name__ == "__main__":
    main()
