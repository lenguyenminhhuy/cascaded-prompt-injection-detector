# Cascade-PID

Two-stage cascade for prompt injection detection: a calibrated small language model (Stage 1) routes uncertain inputs to a stronger guardrail model (Stage 2).

## Quick Start

```bash
# Install
pip install -e .
cp .env.example .env  # fill in WANDB_API_KEY, HF_TOKEN

# End-to-end pipeline
make all

# Or step by step
make data
make train MODEL=qwen2.5-1.5b
make calibrate MODEL=qwen2.5-1.5b
make baselines
make cascade
make sweep
make evaluate
```

## Layout

- `configs/` — YAML configs for data, models, training, evaluation, W&B
- `src/` — library code (data, models, calibration, pipeline, evaluation, utils)
- `scripts/` — entry points for each experiment
- `notebooks/` — exploratory analysis
- `tests/` — pytest suite
- `data/`, `results/` — gitignored outputs

See [cascade-pid-structure.md](../cascade-pid-structure.md) for the full design.

## Experiments

| # | Script | Purpose |
|---|---|---|
| 0 | `download_data.sh` + `download_hf.py` | Fetch source benchmarks (BIPIA, AgentDojo, InjecAgent, HackAPrompt, TensorTrust) |
| 1 | `build_dataset.py` | Build channel-stratified pool + 5 splits (train/cal/test_in_dist/test_cross_channel/test_cross_domain) per `SCHEMA.md` |
| 1b | `audit_dataset.py` / `freeze_dataset.py` | Quality audit (balance, leakage, anti-cheating) then freeze v1 + W&B artifact. See `datasheet.md` |
| 2 | `train_stage1.py` | QLoRA fine-tune Stage 1 candidates |
| 3 | `calibrate.py` | Temperature scaling + threshold search |
| 4 | `run_baselines.py` | PromptGuard 2 + LlamaGuard 4 standalone |
| 5 | `run_cascade.py` | Full pipeline inference |
| 6 | `sweep_thresholds.py` | Detection vs escalation tradeoff curves |

## W&B Project

`cascade-pid` — run groups: `data-prep`, `model-selection`, `calibration`, `baselines`, `cascade-eval`, `threshold-sweep`.
