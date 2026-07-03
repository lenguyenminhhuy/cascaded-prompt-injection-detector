# Baseline Establishment Spec (proposal `prompt_injection_proposal.tex`)

Contract shared by all baseline notebooks/modules. Any agent producing or consuming
baseline artifacts MUST follow this exactly. Paths are relative to
`experiments/cascade-pid/`.

## Goal

Establish the single-stage baselines from the proposal (Sec. Evaluation Protocol →
Baselines) on the proposal's channel-stratified, out-of-distribution eval set
(Table `tab:benchmark`, Evaluation Split column):

| Baseline | What runs | Output type |
|---|---|---|
| PromptGuard 2 (encoder) | `meta-llama/Llama-Prompt-Guard-2-86M` (gated) + ungated fallback `protectai/deberta-v3-base-prompt-injection-v2` | continuous score |
| DataSentinel-7B on every input | known-answer detection, fine-tuned Mistral-7B checkpoint | binary verdict |
| DataSentinel-1B on every input | 1B variant IF a released checkpoint exists; otherwise document unavailability | binary verdict |

(Stage-1-alone requires fine-tuning and is out of scope for baseline establishment.)

## Environment assumptions

Notebooks MUST run in two modes, auto-detected:
- **CUDA** (Colab/GPU server) — primary target for full runs; use 4-bit quantization
  (bitsandbytes) for the 7B model.
- **Apple MPS / CPU** (this machine: M3 Pro, 18 GB RAM) — smoke-test mode only.
  7B must load via 4-bit gguf/ctransformers OR be clearly gated behind
  `if device == "cuda"` with a printed skip message. Never assume bitsandbytes on Mac.

Every notebook has a config cell at top:
```python
RUN_MODE = "smoke"   # "smoke" | "medium" | "full"
```
Per-source example caps: smoke=50, medium=500, full=proposal targets
(conversational 10k, app-structured 10k, document 3k, tool 2k, direct 3k).
All randomness seeded with SEED = 3131.

## File layout

```
data/eval_proposal/eval.jsonl              # the eval set (dataset notebook output)
data/eval_proposal/build_manifest.json     # sizes, sources, seed, RUN_MODE used
results/baselines/predictions/<detector>.jsonl
results/baselines/metrics/<detector>.json
results/baselines/report/                  # comparison tables/figures (report notebook)
notebooks/baselines/00_eval_dataset.ipynb
notebooks/baselines/01_promptguard2.ipynb
notebooks/baselines/02_datasentinel.ipynb
notebooks/baselines/03_baseline_report.ipynb
```
`<detector>` ∈ {`promptguard2_86m`, `protectai_deberta_v2`, `datasentinel_7b`, `datasentinel_1b`}.

## eval.jsonl schema (one JSON object per line)

```json
{
  "id": "opi-000123",              // "<source-abbrev>-<6-digit idx>", stable across runs
  "text": "...",                   // exactly what a detector scores (p || d assembled)
  "label": 1,                      // 0 = benign, 1 = injection
  "category": "injection",         // "conversational" | "application_structured" | "injection"
  "channel": "document",           // injections: "document" | "tool" | "direct"; benign: null
  "source": "openpromptinjection"  // lmsys | dolly | natural_instructions | spp |
                                   // openpromptinjection | agentdojo | struq_synthetic
}
```

## predictions/<detector>.jsonl schema

```json
{
  "id": "opi-000123",
  "label": 1,                  // copied from eval.jsonl
  "channel": "document",       // copied (null for benign)
  "score": 0.97,               // float, HIGHER = more likely injection; null if binary-only
  "pred": 1,                   // detector's hard decision at its native operating point
  "latency_ms": 41.2           // per-input wall-clock (for the cost model k1/k2)
}
```

## Metrics API — `src/evaluation/metrics.py`

```python
def detection_rate_at_fpr(labels, scores, fpr_targets=(0.001, 0.005, 0.01)) -> dict
    # threshold chosen on benign scores; returns {fpr_target: {"dr": .., "threshold": ..}}
def binary_metrics(labels, preds) -> dict          # f1, precision, recall, tpr, fpr, accuracy
def ece(labels, confidences, n_bins=10) -> float
def per_channel(records, metric_fn) -> dict        # metric per channel + overall
def evaluate_detector(pred_path, out_path) -> dict # loads predictions jsonl, writes metrics json
```
Binary-only detectors (DataSentinel): DR@FPR is not applicable — report
`binary_metrics` (its single operating point: TPR and native FPR) and set
`"dr_at_fpr": null` in the metrics json with a `"note"` explaining why.

`metrics/<detector>.json` shape:
```json
{
  "detector": "promptguard2_86m",
  "n": 1234, "n_benign": 900, "n_injection": 334,
  "dr_at_fpr": {"0.001": {...}, "0.005": {...}, "0.01": {...}},
  "binary": {"f1": ..., "tpr": ..., "fpr": ...},
  "per_channel": {"document": {...}, "tool": {...}, "direct": {...}},
  "fpr_benign_only": ...,
  "mean_latency_ms": ...,
  "run_mode": "smoke", "eval_manifest": "data/eval_proposal/build_manifest.json"
}
```

## Notebook conventions

- Pure `pip install` cell at top (commented for local, runnable on Colab).
- Every notebook is standalone-runnable top-to-bottom given `eval.jsonl` exists
  (model notebooks fail fast with a clear message if it doesn't).
- HF auth: read token from env `HF_TOKEN` or `huggingface_hub.login()` cell;
  gated models (PromptGuard 2, Llama) must degrade gracefully to the ungated
  fallback with a printed warning, never crash.
- Write outputs atomically; include a final cell printing the metrics dict.
- Markdown cells cite the proposal section each step implements.

## Notes for consistency with the proposal

- Benign FPR is additionally reported on the benign-only slice (Sec. sizing).
- DataSentinel provenance caveat: it ships with OpenPromptInjection (an eval
  source) — the report notebook must surface this as a limitation flag.
- Latency per input is the cost unit (k1/k2 measurement feeds Eq. cost/breakeven).
