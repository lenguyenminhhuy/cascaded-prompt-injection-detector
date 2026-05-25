.PHONY: setup data train train-all calibrate baselines cascade sweep evaluate all clean

setup:
	pip install -e .
	wandb login
	huggingface-cli login

data:
	bash scripts/download_data.sh
	python scripts/build_splits.py

train:
	python scripts/train_stage1.py --config configs/models/$(MODEL).yaml

train-all:
	python scripts/train_stage1.py --config configs/models/qwen2.5-1.5b.yaml
	python scripts/train_stage1.py --config configs/models/llama3.2-1b.yaml
	python scripts/train_stage1.py --config configs/models/granite-guardian-2b.yaml

calibrate:
	python scripts/calibrate.py --config configs/models/$(MODEL).yaml

baselines:
	python scripts/run_baselines.py

cascade:
	python scripts/run_cascade.py

sweep:
	python scripts/sweep_thresholds.py

evaluate:
	python scripts/evaluate.py

all: data train-all calibrate baselines cascade sweep evaluate

clean:
	rm -rf data/processed data/splits results/
