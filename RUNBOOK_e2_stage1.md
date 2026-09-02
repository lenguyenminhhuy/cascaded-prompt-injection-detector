# E2 Stage-1 model selection — runbook (Kaggle GPU)

Turnkey flow to finish the three-candidate Stage-1 selection without repeating
the mistakes that cost time: P100 (sm_60 incompatible), T4×2 sharding, and the
train+dump-in-one-commit timeout.

Candidates: `qwen2.5-1.5b`, `llama3.2-1b`, `granite-guardian-2b`.

## Hard rules (learned the hard way)

- **Accelerator: `GPU T4 x1`.** Never P100 (CUDA sm_60 — incompatible with this
  PyTorch + bitsandbytes 4-bit). Never T4×2 (`device_map="auto"` shards and
  crashes). If only T4×2 is offered, set `os.environ["CUDA_VISIBLE_DEVICES"]="0"`
  in cell 1 before any torch import.
- **Split training from scoring.** Train with `--no-dump-logits` (commit,
  ~8 h < 12 h wall). Dump logits separately from the saved adapter (inference,
  minutes). The adapter saves *before* any logit dump, so a train-only commit
  cannot lose it.
- **Download artifacts after every run** — a timed-out/failed commit can still
  wipe `/kaggle/working`.
- **Gated models:** `meta-llama/Llama-3.2-1B-Instruct` (and possibly
  granite-guardian) require an `HF_TOKEN`. Add-ons → Secrets → `HF_TOKEN`, then
  `huggingface_hub.login(os.environ["HF_TOKEN"])` in cell 1.

## Selection metric — read this or waste the runs

Val AUROC saturates at ~1.0 for all candidates (in-distribution; sources are
label-pure so a model can win by recognizing dataset style, not injection).
**Rank on `test_cross_domain`** (all `authored`, both labels — defeats the
shortcut), then `test_cross_channel`, then the tail metrics DR@1% FPR and ECE.
Do NOT pick a winner on val.

## Paths (adjust to your Kaggle dataset slugs)

    REPO         = /kaggle/input/datasets/miios18/cascade-pid-repo/cascade-pid
    QWEN_ADAPTER = /kaggle/input/<qwen-output>/results/stage1/qwen2.5-1.5b/adapter
                   (add the prior run's output as an input dataset)

## Cell 1 — session setup (run first, every session)

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"     # single-GPU pin
import sys; REPO = "/kaggle/input/datasets/miios18/cascade-pid-repo/cascade-pid"
sys.path.insert(0, REPO)
import torch; print("torch sees GPUs:", torch.cuda.device_count(), "(want 1)")
# gated models:
from huggingface_hub import login; login(os.environ["HF_TOKEN"])
```

## Step 1 — score the ALREADY-trained qwen adapter (do FIRST; minutes)

Answers "shortcut or real?" before spending hours training. Inline scoring cell
(no dataset re-upload needed — uses src/ already on Kaggle):

```python
from src.models.prompt_template import load_model_config
from src.models.stage1 import Stage1Detector
from src.utils.io import read_jsonl, write_jsonl

def score(cfg_path, adapter, out_dir, splits):
    cfg = load_model_config(cfg_path)
    det = Stage1Detector(cfg, adapter_path=adapter, device="cuda", load_in_4bit=True)
    os.makedirs(out_dir, exist_ok=True)
    for s in splits:
        texts = [(r.get("input") or r.get("text") or "") for r in read_jsonl(s)]  # never printed
        write_jsonl(f"{out_dir}/{os.path.basename(s).replace('.jsonl','')}_logits.jsonl",
                    det.predict(texts))
        print("scored", os.path.basename(s), "->", len(texts), "rows")

SPLITS = [f"{REPO}/data/train_proposal/cal.jsonl",
          f"{REPO}/data/splits/test_in_dist.jsonl",
          f"{REPO}/data/splits/test_cross_channel.jsonl",
          f"{REPO}/data/splits/test_cross_domain.jsonl"]
QWEN_ADAPTER = "/kaggle/input/<qwen-output>/results/stage1/qwen2.5-1.5b/adapter"
score(f"{REPO}/configs/models/qwen2.5-1.5b.yaml", QWEN_ADAPTER,
      "/kaggle/working/results/stage1/qwen2.5-1.5b", SPLITS)
```

Then zip + download `/kaggle/working/results/stage1/qwen2.5-1.5b/*_logits.jsonl`.

## Step 2 — train llama + granite (one train-only COMMIT per model, ~8 h each)

```bash
python scripts/train_stage1.py --config configs/models/llama3.2-1b.yaml \
  --train-file data/train_proposal/train.jsonl \
  --training configs/training_kaggle.yaml \
  --output-dir /kaggle/working/results/stage1/llama3.2-1b \
  --max-samples 12000 --no-dump-logits
```

Same for `granite-guardian-2b`. `--no-dump-logits` is the key change — the run
saves the adapter (`results/stage1/<name>/adapter/`) and stops, well under the
12 h wall. Commit each separately; download each adapter after it finishes.

## Step 3 — score llama + granite adapters (short GPU session, minutes)

Reuse the `score()` cell from Step 1, once per candidate, pointing `--config`
and the adapter path at each. Score the SAME `SPLITS` list (cal + 3 tests) so
the comparison is fair. Download all `*_logits.jsonl`.

## Step 4 — rank locally (NO GPU, on your Mac)

Put every candidate's logit dumps under `results_kaggle/stage1/<name>/`, then:

```bash
# per-candidate detail (val already done for qwen):
PYTHONPATH=. python scripts/score_stage1_logits.py \
  --logits results_kaggle/stage1/qwen2.5-1.5b/test_cross_domain_logits.jsonl \
  --split  data/splits/test_cross_domain.jsonl

# full comparison table across all candidates x splits, ranked on cross-domain:
PYTHONPATH=. python scripts/rank_stage1.py --results-dir results_kaggle/stage1
```

`rank_stage1.py` picks the winner by `test_cross_domain` accuracy and prints
AUROC / acc / DR@1% / ECE per split per candidate.

## GPU budget (30 h/week)

- Step 1 (qwen scoring): ~0.1 h
- Step 2 (train ×2): ~16 h
- Step 3 (score ×2): ~0.2 h
Total ≈ 16.5 h — one week, with margin for one failed run.

## Artifact convention (per candidate, under results/stage1/<name>/)

    adapter/                 trained LoRA + tokenizer
    train_summary.json       hyperparams, final loss, steps, device
    val_logits.jsonl         per-row {logp_benign, logp_injection, p_safe}
    cal_logits.jsonl
    test_in_dist_logits.jsonl
    test_cross_channel_logits.jsonl
    test_cross_domain_logits.jsonl