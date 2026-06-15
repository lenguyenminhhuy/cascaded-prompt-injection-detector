"""Experiment 1: build train/val/cal/eval JSONL splits per Table 2."""

import argparse
import logging
from pathlib import Path

from src.data.loaders.alpaca import AlpacaLoader
from src.data.loaders.bipia import BipiaLoader
from src.data.loaders.dolly import DollyLoader
from src.data.loaders.hackaprompt import HackapromptLoader
from src.data.loaders.ifeval import IFEvalLoader
from src.data.loaders.injecagent import InjecAgentLoader
from src.data.loaders.lmsys_chat import LmsysChatLoader
from src.data.loaders.natural_instructions import NaturalInstructionsLoader
from src.data.loaders.notinject import NotInjectLoader
from src.data.loaders.open_prompt_injection import OpenPromptInjectionLoader
from src.data.loaders.spp import SppLoader
from src.data.loaders.struq import StruqLoader
from src.data.loaders.ultrachat import UltrachatLoader
from src.data.loaders.wildguard import WildguardLoader
from src.data.split_builder import SplitBuilder
from src.utils.io import load_yaml
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val/cal/eval JSONL splits")
    parser.add_argument("--splits", default="configs/data/splits.yaml")
    parser.add_argument("--out", default="data/splits")
    parser.add_argument("--injecagent-dir", default="../datasets/InjecAgent/data")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Per-loader sample cap (for fast test runs)")
    args = parser.parse_args()

    splits_cfg = load_yaml(args.splits)
    seed = splits_cfg.get("seed", 42)
    set_seed(seed)

    log.info("seed=%d, out=%s", seed, args.out)

    mx = args.max_samples

    train_loaders = [
        BipiaLoader(max_samples=mx),
        InjecAgentLoader(data_dir=args.injecagent_dir, split="train", max_samples=mx),
        HackapromptLoader(max_samples=mx),
        StruqLoader(max_samples=mx),
        UltrachatLoader(max_samples=mx),
        IFEvalLoader(max_samples=mx),
        AlpacaLoader(max_samples=mx),
        NotInjectLoader(max_samples=mx),
    ]

    eval_loaders = [
        OpenPromptInjectionLoader(max_samples=mx),
        InjecAgentLoader(data_dir=args.injecagent_dir, split="eval", max_samples=mx),
        WildguardLoader(max_samples=mx),
        LmsysChatLoader(max_samples=mx),
        DollyLoader(max_samples=mx),
        NaturalInstructionsLoader(max_samples=mx),
        SppLoader(max_samples=mx),
    ]

    ratios = splits_cfg.get("train_pool", {}).get("ratios", {})
    balance_cfg = splits_cfg.get("balance", {})

    builder = SplitBuilder(
        train_loaders=train_loaders,
        eval_loaders=eval_loaders,
        out_dir=args.out,
        seed=seed,
        train_ratio=ratios.get("train", 0.80),
        val_ratio=ratios.get("val", 0.10),
        target_injection_ratio=balance_cfg.get("target_injection_ratio", 0.5),
    )

    counts = builder.build()
    log.info("Done: %s", counts)
    Path(args.out).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
