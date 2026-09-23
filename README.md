# Confidence-Based Two-Stage Cascading for Prompt-Injection Detection

Research code for a two-stage prompt-injection detector. A small fine-tuned
language model (Stage 1) screens every request and clears only the inputs it
judges safe with high confidence. Everything else is deferred to a larger
task-specific detector (Stage 2), which makes every blocking decision.

The design is deliberately one-sided: Stage 1 can allow, never deny. This keeps
all blocking behaviour with the stronger model while removing the confidently
benign traffic from the expensive path.

This repository contains the training, evaluation, and latency-measurement code
behind the accompanying paper.

## What is here

| Path | Contents |
|---|---|
| `src/data/` | Benchmark construction, deduplication, contamination screening |
| `src/models/` | Stage-1 and Stage-2 fine-tuning (QLoRA over 4-bit base weights) |
| `src/pipeline/` | Cascade routing and the two-stage decision flow |
| `src/calibration/` | Threshold selection and temperature scaling |
| `src/evaluation/` | Metrics, cost model, latency profiling |
| `scripts/data/` | Download sources, build, dedupe and freeze the splits |
| `scripts/train/` | Fine-tune Stage 1 and Stage 2, calibrate, dump logits |
| `scripts/eval/` | Detection, routing and robustness results |
| `scripts/cost/` | Latency measurement and the cost model |
| `scripts/figures/` | Figures in the paper |
| `configs/` | Model, training, and experiment configuration |
| `tests/` | Unit tests |
| `datasheet.md` | Description of the benchmark and its sources |
| `REPRODUCE.md` | Which script produces which table and figure |

## Requirements

- Python 3.10
- An NVIDIA GPU. The reported latency figures were measured on a single A10G
  (24 GB class) at batch size 1. Stage-2 fine-tuning needs at least 16 GB.

```bash
pip install -r requirements.txt
```

Key dependencies: `torch>=2.1`, `transformers>=4.40`, `peft>=0.11`,
`bitsandbytes>=0.43`, `datasets>=2.19`, `trl>=0.8`, `scikit-learn>=1.4`.

## Data

The evaluation benchmark and the development corpus are assembled from publicly
available datasets. Each source keeps the licence of its original provider, so
the assembled splits are not redistributed here. `datasheet.md` lists every
source and how the splits are built; `scripts/data/download_data.sh` and
`scripts/data/build_dataset.py` rebuild them.

## Quick start

```bash
make setup        # install and verify the environment
make data         # download sources and build the splits
make train-all    # fine-tune the Stage-1 candidates and Stage-2
make calibrate    # fit and freeze the routing threshold on the calibration split
make eval         # run the end-to-end cascade and write the paper metrics
make figures      # draw the figures
```

`make all` runs the full sequence. See `REPRODUCE.md` for per-table commands.

## Models

Stage 1 is evaluated with three candidates of roughly 1–2.5B parameters
(Llama 3.2-1B-Instruct, Qwen2.5-1.5B-Instruct, Granite Guardian 3.0-2B).
Stage 2 is fine-tuned from Mistral-7B-v0.1. All are trained with QLoRA,
updating low-rank adapters over frozen 4-bit base weights.

## Licence

MIT. See `LICENSE`. The source datasets remain under their own licences.
