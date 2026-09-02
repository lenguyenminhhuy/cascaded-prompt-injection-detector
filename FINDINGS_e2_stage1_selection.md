# E2 — Stage-1 model selection: findings

_Last updated: 2026-08-18; status and next-steps refreshed 2026-09-02. Scope: QLoRA
fine-tuning + scoring of the three Stage-1 candidates (Qwen2.5-1.5B, Llama-3.2-1B,
Granite-Guardian-2B) and the model-selection comparison._

> Payload hygiene: all numbers here are aggregates (counts, metrics). No dataset
> text. See `RUNBOOK_e2_stage1.md` for the operational Kaggle flow.

## 1. Status at a glance

| Candidate | Base model | Trained | Scored |
|---|---|---|---|
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B-Instruct` | ✅ 12k | ✅ val/cal/3 tests |
| Llama-3.2-1B | `meta-llama/Llama-3.2-1B-Instruct` | ✅ 12k | ✅ val/cal/3 tests |
| Granite-Guardian-2B | `ibm-granite/granite-guardian-3.0-2b` | ✅ 9k | ✅ val/cal/eval/3 tests (2026-08-20) |

All three candidates are trained and scored; artifacts under `results_kaggle/stage1/<name>/`,
verified locally per the cloud-GPU handoff protocol (row counts match val=5,724 / cal=5,722 /
eval=25,747, metrics parse, no NaN).

**Selection is settled** (`results/metrics/stage1_selection.json`, filled 2026-08-20). Two rules
give two answers, and both are reported:

- The **saturated-validation rule** (val DR@1%FPR primary, val ECE tie-break) cannot separate the
  candidates — all three hit val DR ≈ 1.000 — so it falls through to ECE and names
  **Granite-Guardian-2B** (0.0006 < 0.0007 < 0.0014). That is a coin-flip on a saturated metric,
  not a finding.
- On the **25,747-row OOD eval set**, which does separate them, **Qwen2.5-1.5B wins**
  (AUROC 0.9288 / DR@1%FPR 0.1710 / ECE 0.2591) ahead of Granite (0.9050 / 0.1133 / 0.2908) and
  Llama (0.8815 / 0.1889 / 0.3470). Note Llama has the best eval DR@1%FPR but the worst AUROC and
  ECE, so the winner still depends on the metric.

The thesis records Qwen as the selection and states plainly that it is a test-set selection.
Granite was not carried into the cascade; **Qwen and Llama are both evaluated end-to-end** so the
Stage-1 choice is reported as a two-arm comparison rather than a single pick (see
`FINDINGS_E5_E7_cascade_headline.md`).

## 2. Training runs (shared recipe)

Recipe (all candidates, Kaggle **T4 ×1**): QLoRA r=16 / α=32, 4-bit NF4, bf16,
lr 2e-4 cosine, **1 epoch**, max seq 1024, effective batch 32 (per-device 4 ×
grad-accum 8). Pinned stack: `trl==1.10.0`, `transformers>=4.56,<5`, `peft 0.19.1`.

| Candidate | n_train | steps | final train loss | notes |
|---|---|---|---|---|
| Qwen2.5-1.5B | 12,000 | 356 | 0.0974 | — |
| Llama-3.2-1B | 12,000 | 358 | 0.0166 | tightest fit (see §4) |
| Granite-Guardian-2B | 9,000 | 266 | 0.0668 | reduced subset (timeout fix, §5); q/k/v/o LoRA only |

Train loss is **not** a quality signal — a lower loss on a label-pure corpus can mean
tighter style-fitting, not better detection. Verdict deferred to OOD scoring.

## 3. Provisional 2-way comparison (Qwen vs Llama)

Scored on the small `data/splits/*` splits. **Superseded 2026-08-20:** Granite was
scored on the full 25,747-row `eval_proposal` set; the 3-way result is in
`results/metrics/stage1_selection.json` (Granite AUROC 0.905 / DR@1%FPR 0.1133 /
ECE 0.2908). This section is retained as the state on 2026-08-18 only. Ranking split per
runbook = `test_cross_domain`. Split label balance: cross_domain 149 inj / 104 benign
(n=253); cross_channel 183/183 (n=366); in_dist 78 inj / 445 benign (n=523).

| Split | metric | Llama-3.2-1B | Qwen2.5-1.5B |
|---|---|---|---|
| val | AUROC / DR@1% / ECE | 1.000 / 0.999 / 0.0014 | 1.000 / **1.000** / **0.0007** |
| cal | AUROC / DR@1% / ECE | 1.000 / 1.000 / 0.0014 | 1.000 / 1.000 / **0.0003** |
| test_in_dist | AUROC / ECE | 0.684 / 0.528 | **0.818** / **0.484** |
| test_cross_channel | AUROC / DR@1% / ECE | 0.853 / 0.448 / 0.317 | **0.896** / **0.492** / **0.270** |
| test_cross_domain | AUROC / DR@1% / ECE | **0.971** / **0.920** / 0.411 | 0.931 / 0.228 / **0.376** |

`rank_stage1.py` verdict: **Llama** (by cross_domain AUROC, ECE tie-break). This is
**not** a safe conclusion — see §4.

## 4. Key insights

1. **Validation saturates for every candidate** (AUROC 1.000, ECE ~0.001 on val+cal).
   Val cannot discriminate the models — selection must rest on OOD splits. **3-of-3 confirmed:**
   Granite also lands at val AUROC 1.000 / DR@1%FPR 1.000 / ECE 0.0006.
2. **OOD calibration collapse.** On OOD splits both models squash `p_safe` toward 0
   (Llama cross_domain `p_safe` max = 0.0001) and default to "injection"; ECE jumps to
   0.27–0.53 vs ~0.001 in-dist. **But discrimination survives** — AUROC 0.85–0.97 — so
   the ranking signal is intact; only the operating point is broken. This is a
   calibration problem, not a capability failure, and it is the concrete motivation for
   E3 temperature scaling.
3. **The winner flips by split.** On AUROC: Llama wins cross_domain; Qwen wins
   cross_channel AND in_dist. Qwen is the more consistent OOD generalizer; Llama peaks
   on one axis. "Llama wins" is an artifact of ranking on a single split.
4. **The selection metric determines the winner.** The thesis rule (val DR@1%FPR →
   ECE tie-break) would pick **Qwen** of these two (ECE 0.0007 < 0.0014). The runbook rule
   (cross_domain AUROC) picks **Llama**. **Reconciled on eval data (§1):** with Granite in the
   field the saturated-val rule falls through to Granite, while eval AUROC picks Qwen — and eval
   DR@1%FPR picks Llama. The metric still determines the winner; the thesis reports Qwen and Llama
   end-to-end rather than resolving it by fiat.
5. **Not a shortcut artifact.** Both models retain real OOD signal (AUROC well above
   chance on unseen sources), supporting the cascade premise that a fine-tuned SLM
   Stage-1 is worth building.

## 5. Operational findings (things that cost time)

- **Kaggle env drift broke training.** Kaggle now ships `transformers 5.0.0` + latest
  `trl 1.10.0`, which are incompatible: trl imports `is_torch_distributed_available`
  (removed in transformers 5) → ImportError, and its `SFTConfig` inheritance breaks →
  `warmup_ratio` TypeError. **Fix: pin `transformers>=4.56,<5`** (trl 1.10.0 is a
  transformers-4.x release). Baked into the notebook's cell 1 (`e2_stage1_kaggle.ipynb`).
  `-U` latest must NOT be used.
- **Granite-2B hit the 12h Kaggle commit wall at 12k samples.** Log analysis: genuinely
  compute-bound at ~120 s/it (1.5× the 1B/1.5B), good T4 utilization (not starved),
  setup cheap (~150s). 12k = ~354 steps ≈ 11.8h + saves → over the wall.
  **Fix: `MAX_SAMPLES=9000`** → 266 steps; measured run finished ~10.2h (~1.7h buffer).
  Rate crept to ~153 s/it late (likely thermal throttle) but the buffer absorbed it.
- Scoring (`score_adapter`) also needs a **short** GPU session — 4-bit is CUDA-only —
  but only minutes, run interactively, not a commit.

## 6. Caveats limiting the current numbers

- **Splits are tiny and under-powered** (253/366/523). `score_stage1_logits.py` flags
  DR@0.1%/0.5% as unresolvable on ~104 benign rows; the DR@1% gaps are noisy.
- **`data/splits/test_*` may be the retired v1 dataset** (`EXPERIMENTS.md §5`). Verify
  what these splits are before ranking a winner on them.
- **Selection-grade decision belongs on `eval_proposal` (25,747 rows) post-E3**, where
  DR@1%FPR is actually resolvable — not on these splits.
- **Granite fairness footnote:** trained on 9k vs 12k for the others. All candidates are
  well past the ~0.3-epoch saturation point, so selection is unaffected.
- **The planned full-corpus re-train never happened.** The intent was to re-train the selected
  Stage-1 on all 45,349 rows (3 epochs, seq 2048) before the cascade. No such run exists: the
  only Stage-1 `train_summary.json` artifacts report `n_train` 12,000 (qwen, llama) and 9,000
  (granite), and the adapters the cascade was scored with are those same ones. **Every Stage-1
  number in the thesis therefore comes from a 12,000-row, 1-epoch, seq-1024 adapter** — the
  `sec:exp-stage1` clause promising the full-corpus re-train (§7 below) must be removed or the
  run must actually be done. Stage 2 (M2) *was* trained on the full corpus (50,349 rows).

## 7. Thesis edits made this session

File: `../../03_thesis_writing/main_31July2026_RWrevised.tex`

- **line ~750** (`\subsection{Cascade detection and cost}`, `sec:exp-stage1` placeholder):
  hyperparameters corrected to the actual regime (first 12,000 examples, 1 epoch, seq
  1024, effective batch 32) + clause that the winner is re-trained on the full 45,349 /
  3 epochs / 2048 for the cascade.
- **line ~443** (`sec:stage1select`): selection metric reconciled to **DR@1%FPR primary
  → ECE → latency**, removing the earlier F1-vs-DR@1%FPR contradiction with the placeholder.
- **Deferred:** the "val saturates → decide on OOD splits" rewrite of `sec:stage1select`
  waited on (a) 3-of-3 saturation confirmation and (b) verifying what the `data/splits` actually
  are. (a) is now confirmed — Granite also saturates at val DR 1.000 / ECE 0.0006 — so the rewrite
  is unblocked on that side; (b) is still open.
- **Now known to be wrong:** the clause added at line ~750 saying the winner is re-trained on the
  full 45,349 / 3 epochs / 2048 for the cascade. That run was never done (see §6). Delete the
  clause; the cascade uses the 12,000-row adapters.

## 8. Next steps

1. ~~**Score Granite**~~ — **DONE 2026-08-20.** All six splits scored; eval = 25,747 rows.
2. ~~**Run the 3-way comparison**~~ — **DONE**, via `scripts/fill_stage1_selection.py` →
   `results/metrics/stage1_selection.json`. Note its `"winner"` field reads
   `granite-guardian-2b` on the saturated-validation rule; the thesis selects Qwen on
   the eval set and records that as test-set selection.
3. **Verify the `data/splits` provenance** (v1-retired vs valid v2 selection splits) — **still
   open.** This is the one caveat in §6 that nothing since has cleared. It only affects §3's
   provisional 2-way table, since selection now rests on `eval_proposal`.
4. ~~**Real selection** on `eval_proposal` (25,747)~~ — **DONE 2026-08-20**, all three candidates
   scored on the full eval set; results in `results/metrics/stage1_selection.json` and summarised
   in §1. Ranking there uses raw `p_safe`; the E3 temperature-scaled variant was not used for
   selection because AUROC (the discriminating metric) is invariant to a monotone rescaling.
5. **Fill the `sec:exp-stage1` selection table and do the deferred `sec:stage1select` rewrite** —
   still open, and now also needs the corrections in §6/§7 (drop the full-corpus re-train clause).
6. **Remove or execute the full-corpus Stage-1 re-train** (§6). Removing the claim is the cheap
   option and costs nothing empirically; executing it is one Kaggle training run per model.
7. Optional infra: add `--resume-from-checkpoint` to `scripts/train_stage1.py` so a
   wall-timeout becomes resumable instead of lost work.
