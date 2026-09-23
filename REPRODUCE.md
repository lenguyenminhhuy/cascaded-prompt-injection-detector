# Reproducing the paper

Every table and figure below maps to the script that produces it. Run
`make data` first; the training and evaluation steps assume the frozen splits
exist.

Paths are relative to the repository root. Outputs land under `results/`,
which is not tracked in git.

## Prerequisites

```bash
pip install -r requirements.txt
make data          # download sources, build and freeze the splits
make train-all     # fine-tune the three Stage-1 candidates and Stage-2
make calibrate     # fit and freeze theta_safe on the calibration split
```

## Tables

| Paper table | What it reports | Command |
|---|---|---|
| Dataset sources and counts | Benchmark composition | `python scripts/build_dataset.py` then `python scripts/audit_dataset.py` |
| Detector latency by sequence length | Per-input latency at eight reference lengths | `python scripts/benchmark_latency_curve.py` then `python scripts/merge_latency_curves.py` |
| Stage-1 routing performance | AUROC and ECE for the three Stage-1 candidates | `python scripts/score_stage1_logits.py` then `python scripts/rank_stage1.py` |
| Example paths through the cascade | Trace rows | `python scripts/inspect_samples.py` |
| End-to-end out-of-distribution results | Headline cascade result | `python scripts/eval_cascade.py` |
| In-distribution detection and routing | Cascade on the validation split | `python scripts/eval_cascade_indist.py` |
| Cost reduction at 512 tokens | Break-even and measured savings | `python scripts/measure_latency.py` |
| In-distribution cost and latency | Length-weighted cost, latency quantiles | `python scripts/measure_latency_indist.py` then `python scripts/analyze_indist_latency_measured.py` |
| Cascade detection by channel | Per-channel detection and auto-passes | `python scripts/eval_cascade.py` (per-channel breakdown is in the summary output) |
| Evaluation AUROC by input length | Band-wise ranking | `python scripts/analyze_length_weighted_cost.py` |
| Effect of adding long benign examples | Original against length-balanced training data | see **Known gap** below |
| Injected-instruction overlap by channel | Tail 5-gram coverage | `python scripts/diag_direct_overlap.py` and `python scripts/diag_seen_vs_unseen.py` |
| Detection after changing only the payload | Reword manipulation | `python scripts/build_e14_reword.py`, `python scripts/verify_e14_unseen.py`, `python scripts/analyze_e14.py` |
| Fine-tuning configurations | Training runs | `python scripts/train_stage1.py --config configs/models/<model>.yaml` |

## Figures

| Paper figure | Command |
|---|---|
| Failure overlap | `python scripts/analyze_failure_overlap.py` then `python scripts/make_figure_failure_overlap.py` |
| Detection and cost against escalation rate | `python scripts/sweep_theta_safe.py` then `python scripts/make_figure_theta_sweep.py` |
| In-distribution against out-of-distribution detection | `python scripts/make_figures.py` |
| Length profile | `python scripts/make_figure_length_profile.py` |

## Supporting analyses

| Analysis | Command |
|---|---|
| Threshold transfer across distributions | `python scripts/sweep_thresholds.py` |
| Temperature-scaling calibration diagnostic | `python scripts/calibrate.py` |
| Contamination and duplicate screening | `python scripts/cross_eval_dedup.py`, `python scripts/recheck_contamination.py` |
| Two-sided routing comparison | `python scripts/eval_cascade_twosided.py` |
| Stage-1 score sensitivity | `python scripts/sensitivity_stage1_noise.py` |
| Precision ablation (4-bit against 16-bit) | `bash scripts/run_precision_ablation.sh` |

## Known gap

The benign-length ablation compares Stage-1 trained on the original data
against Stage-1 trained on a length-balanced variant, in which short benign
rows are replaced by long benign documents. The variant is built by
`scripts/build_length_balanced_trainset.py` and trained with
`scripts/train_stage1.py` on the resulting split. The analysis step that turns those runs into the reported
correlation, AUROC, and band-wise detection numbers is not yet scripted in this
repository. The trained adapters and evaluation logits are retained, so the
analysis can be regenerated, but there is currently no single command for it.

## Measurement notes

Latency figures are single-request, batch size 1, on one NVIDIA A10G (24 GB
class). Each measurement takes the median of 40 timed repetitions after eight
warm-up passes. Reported cost ratios and break-even points are measurements for
that setup, not hardware-independent constants.

Sampling and splitting use fixed seeds (3131 and 42).
