"""Experiment 2: QLoRA fine-tune a Stage 1 candidate."""

import argparse

from src.utils.io import load_yaml
from src.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to model config YAML")
    parser.add_argument("--training", default="configs/training.yaml")
    args = parser.parse_args()

    model_cfg = load_yaml(args.config)
    train_cfg = load_yaml(args.training)
    log.info("training %s", model_cfg["name"])
    log.info("epochs=%s lr=%s", train_cfg["trainer"]["num_train_epochs"], train_cfg["optimizer"]["lr"])

    # TODO: load base model in 4-bit, attach LoRA adapters, run SFTTrainer.
    raise NotImplementedError("train_stage1: implement QLoRA pipeline")


if __name__ == "__main__":
    main()
