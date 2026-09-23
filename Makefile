.PHONY: setup data audit freeze train train-all calibrate eval figures all clean

setup:
	pip install -e ".[train]"
	wandb login
	huggingface-cli login

data:
	bash scripts/data/download_data.sh
	python scripts/data/download_hf.py
	python scripts/data/build_dataset.py
	python scripts/data/audit_dataset.py
	python scripts/data/freeze_dataset.py --version v1
	python scripts/data/build_stage2_trainset.py

audit:
	python scripts/data/audit_dataset.py

freeze:
	python scripts/data/freeze_dataset.py --version v1

train:
	python scripts/train/train_stage1.py --config configs/models/$(MODEL).yaml

train-all:
	python scripts/train/train_stage1.py --config configs/models/qwen2.5-1.5b.yaml
	python scripts/train/train_stage1.py --config configs/models/llama3.2-1b.yaml
	python scripts/train/train_stage1.py --config configs/models/granite-guardian-2b.yaml

calibrate:
	python scripts/train/calibrate.py --config configs/models/$(MODEL).yaml

# Scoring a trained adapter to logits needs per-run arguments; see REPRODUCE.md.
eval:
	python scripts/eval/score_stage1_logits.py
	python scripts/eval/rank_stage1.py
	python scripts/eval/sweep_theta_safe.py
	python scripts/eval/eval_cascade.py

figures:
	python scripts/figures/make_figures.py

all: data train-all calibrate eval figures

clean:
	rm -rf data/processed data/splits data/audit data/pool.jsonl \
	  data/raw_positives.jsonl data/coverage_report.json data/VERSION.json results/
