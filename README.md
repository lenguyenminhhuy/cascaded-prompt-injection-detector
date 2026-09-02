# Cascade-PID — Cascaded Prompt Injection Detection with Small Language Models

A two-stage cascade for prompt-injection detection: a QLoRA-fine-tuned **small language model
(Stage 1)** acts as a cheap one-sided pre-filter that auto-passes traffic it is confident is
benign, and escalates everything else to a **stronger 7B detector (Stage 2)**. The research
question is not "can we build a better detector" — it is *"can a strong detector be made
affordable at scale without losing detection quality?"*

Research code for an RMIT University thesis / journal submission (Huy Le, `s3777280@rmit.edu.vn`).

---

## 1. Executive summary

Prompt injection is OWASP's top risk for LLM applications, and the field's detectors split into
two unsatisfying camps: small encoder classifiers that are cheap but brittle, and 7B+ guard models
that are accurate but too expensive to run on every input. The result is that the highest-throughput
deployments — the ones most exposed — are defended by the weakest detectors.

This project builds a purpose-built **out-of-distribution benchmark** (25,747 inputs; 7,113 attacks
/ 18,634 benign; training and evaluation sources strictly disjoint; stratified by the *channel* an
injection arrives through) and reports four things on it:

**1. Every affordable single-stage detector fails out-of-distribution.** Encoder classifiers that
publish ~97.5% recall land at **22–29%** detection at a 1% false-positive rate. Fine-tuned 1–2B
SLMs used alone reach only **11–19%**. The one usable strong detector is **M2**, our own Mistral-7B
QLoRA classifier, at **40.2%**.

**2. The released DataSentinel-7B guard is unusable here, for a mechanistic reason.** Its
known-answer detection flags an input when a planted canary instruction is not echoed back — so
*any* benign text containing instructions the model follows displaces the canary and becomes a
false positive. It flags **65.1% of benign traffic**, rising with input length (0.29 → 0.89 across
length quartiles). It detects instruction *presence*, not malicious *intent*. That is why Stage 2
is a classifier we trained on the *same base weights* (Mistral-7B-v0.1) rather than the released
checkpoint — holding the base model fixed makes it a controlled comparison of training objectives.

**3. The cascade keeps Stage-2 detection while paying for Stage 2 on ~40–52% of traffic.** With the
routing threshold frozen on the calibration split, Llama-3.2-1B → M2 escalates 51.6% of inputs and
**matches** M2-on-every-input (gap CI straddles zero); Qwen-2.5-1.5B → M2 escalates 39.7% and
**exceeds** it (gap CI entirely above zero). The gain is real, not a selection artifact: auto-passing
easy benign traffic concentrates the fixed 1% false-positive budget onto a smaller, harder escalated
pool, letting Stage 2 run a looser threshold. Measured inference cost falls **26–37%** when every
request is charged at 512 tokens, and **17–20%** on the benchmark's real (much shorter) length mix —
provided Stage 1 runs in bf16.

**4. Two bounds must travel with that headline.** *(a)* The shared ~0.40 detection level is a
composition, not a level: **direct injection is nearly solved (0.92) and document-embedded/indirect
injection is almost entirely missed (0.05)** — by *both* stages. Remove the document channel and
everything jumps to 0.63–0.65. The cascade cheaply resolves the easy channel; neither stage handles
the hard one. *(b)* Every "at 1% FPR" number here — cascade **and** baselines — uses an
**eval-set-calibrated threshold**. Freeze that threshold on the calibration split and apply it
out-of-distribution and you get **57.9% FPR, not 1%**. Operating points do not transfer OOD; that is
a first-order open problem, reported as such rather than buried.

The honest contribution: **an architecture that makes a strong detector affordable at scale without
losing accuracy, plus a map of where the whole field still fails** (indirect injection) and why the
standard 1%-FPR evaluation is optimistic (thresholds don't transfer). The negative results are part
of the finding, not a gap in it.

---

## 2. How it works

### 2.1 Architecture

```
                       ┌─────────────────────────────┐
   p ‖ d  ──────────►  │ Stage 1: fine-tuned SLM     │
  (prompt ‖ untrusted  │ Llama-3.2-1B / Qwen-2.5-1.5B│
   data)               │ QLoRA, 2-forward scoring    │
                       └──────────┬──────────────────┘
                                  │  p_safe
                    ┌─────────────┴──────────────┐
       p_safe ≥ θ   │                            │  p_safe < θ
     (auto-pass)    ▼                            ▼   (escalate)
              ┌──────────┐            ┌─────────────────────────┐
              │  ALLOW   │            │ Stage 2: M2             │
              │ (no S2   │            │ Mistral-7B-v0.1 + QLoRA │
              │  call)   │            │ continuous score        │
              └──────────┘            └───────────┬─────────────┘
                                                  │ ≥ θ₂ → BLOCK
                                                  └────────────► ALLOW / BLOCK
```

Three design decisions carry the result:

- **The cascade is one-sided, and that is an empirical finding rather than an assumption.** Stage 1
  only auto-*passes*; it never blocks. A two-sided Stage 1 with a `DENY` branch is never
  budget-positive here (E8): an auto-blocked benign input spends the global 1% false-positive budget
  that Stage 2 would have spent more accurately, and Stage 1 is too miscalibrated OOD to block
  safely. So all cost savings come from passing benign traffic, and all end-to-end false negatives
  come from auto-passed attacks.
- **Routing is on confidence rank, not on calibration.** Detection rate at a fixed FPR is rank-based
  and therefore invariant to monotone rescaling, so temperature scaling is reported as a diagnostic
  (E3) and the frontier is computed on raw scores. Calibration is not the mechanism — routing is.
- **The routing threshold θ_safe is selected on the calibration split and frozen** before evaluation,
  so no number below comes from a test-set argmax. Stage 2's threshold is set on eval benign to hit
  1% FPR overall — the same convention the single-stage baselines use, so each system sets exactly
  one eval threshold and the comparison stays apples-to-apples. Confidence intervals are 1,000×
  paired bootstrap.

### 2.2 Cost model

Per-input cost is `C(e) = k₁ + e·k₂`, where `k₁` is the Stage-1 call, `k₂` the Stage-2 call, and `e`
the escalation rate. The cascade is cost-positive while `e < 1 − k₁/k₂`, i.e. below the **break-even
escalation** `e*`. Both `k₁` and `k₂` are **measured**, not estimated: `scripts/benchmark_latency_curve.py`
times eight prompt lengths (64–2048 tokens) on an NVIDIA A10G, batch = 1, median of 40 reps, model
load and warm-up excluded, in the same 2-forward constrained-decode regime used to produce the logits.

### 2.3 The benchmark

Two corpora, built to be disjoint by source so the evaluation is genuinely out-of-distribution
(`scripts/build_dataset.py`, `scripts/cross_eval_dedup.py`; asserted in
`notebooks/01_data_exploration.ipynb`).

**Training** (`data/train_proposal/`) — 56,795 rows: train 45,349 / val 5,724 / cal 5,722. Sources:
alpaca, ultrachat, ifeval, notinject, bipia, injecagent, hackaprompt, struq_generated,
synthetic-encoded. Stage 2 additionally uses `train_stage2.jsonl` (50,349 rows) with a long-benign
augmentation block that removes the "long text ⇒ injection" length shortcut latent in the base
corpus (benign median 72 chars vs injection median 868).

**Evaluation** (`data/eval_proposal/eval.jsonl`) — 25,747 rows, **zero source overlap with training**:

| slice | source | n |
|---|---|---|
| conversational benign | lmsys | 9,452 |
| app-structured benign | dolly + natural_instructions | 4,891 + 4,291 |
| **document** injection | openpromptinjection (7 tasks × 5 attacks) | 2,819 |
| **tool** injection | agentdojo (4 suites) | 1,464 |
| **direct** injection | struq_synthetic (reserved links, seed-42 partition) | 2,830 |
| **total** | | **25,747** (18,634 benign / 7,113 injection) |

MinHash-LSH + MiniLM cross-eval dedup removed 295 near-duplicates (26,042 → 25,747); residual
leakage audit is clean (max cosine 0.899 < 0.95, max Jaccard 0.487 < 0.5).

> **Payload hygiene.** `data/` holds live jailbreak and injection text. Inspect it with
> `scripts/inspect_samples.py` (prints id / source / channel / label / length / sha256 — never the
> text), and never paste dataset rows into an AI-assistant session. See `CLAUDE.md`.

---

## 3. Results

All numbers: 25,747-row OOD benchmark, detection rate at 1% false-positive rate (DR@1%FPR).
Every plotted value and its provenance is in `results/figures/thesis/figure_data.json`, produced by
the same metric code as `scripts/eval_cascade.py`, so figures and tables cannot disagree.

### 3.1 Single-stage detectors all fail out-of-distribution

![In-distribution vs out-of-distribution detection](docs/figures/generalization_gap_double.png)

*In-distribution detection (validation split, or the vendor's own reported figure) against detection
on this OOD benchmark. Everything that saturates in-distribution collapses OOD. PromptGuard 2's
in-distribution bar is Meta's reported 97.5% recall at 1% FPR; ProtectAI reports F1 = 0.95 on its own
eval set rather than DR@1%FPR, hence `n/a`.*

| detector | params | AUROC | **DR@1%FPR** | eval ECE |
|---|---|---|---|---|
| PromptGuard 2 (off-the-shelf) | 86M | 0.822 | 0.221 | — |
| ProtectAI DeBERTa-v3 (off-the-shelf) | DeBERTa-v3-base | 0.863 | 0.293 | — |
| Granite-Guardian-2B (ours, QLoRA) | 2B | 0.905 | 0.113 | 0.291 |
| Stage 1 · Llama-3.2-1B (ours, QLoRA) | 1.24B | 0.881 | 0.189 | 0.347 |
| Stage 1 · Qwen-2.5-1.5B (ours, QLoRA) | 1.54B | 0.929 | 0.171 | 0.259 |
| **Stage 2 · M2 Mistral-7B (ours, QLoRA)** | 7.24B | **0.934** | **0.402** | 0.204 |
| DataSentinel-7B (released checkpoint) | 7.24B | — | verdict-only: TPR 0.735 at **FPR 0.651** | — |

![ROC in the low-FPR region](docs/figures/roc_lowfpr_double.png)

*The same detectors on a log FPR axis — the region the 1%-FPR headline actually lives in. AUROC hides
this: ProtectAI's 0.863 AUROC and PromptGuard 2's 0.822 both translate to under 30% detection at the
operating point a production deployment would have to use.*

### 3.2 Why Stage 2 is a model we trained, not the released guard

![DataSentinel-7B false-positive rate by input length](docs/figures/datasentinel_length_double.png)

*DataSentinel-7B's benign false-positive rate by input-length quartile (quartile edges 61 / 219 /
1201 characters; n ≈ 4.6k per bin): **0.29 → 0.67 → 0.75 → 0.89**, overall 0.651. The failure scales
with length because longer benign text contains more instructions to displace the canary — a
mechanism failure, not a capacity one. M2 is trained on the same Mistral-7B-v0.1 base weights, so the
difference isolates the training objective, and unlike a binary verdict it emits a continuous score,
which the cascade needs for routing.*

### 3.3 Headline — the cascade matches Stage 2 on 40–52% of the traffic

![Accuracy-cost frontier and measured cost reduction](docs/figures/cost_frontier_double.png)

*(a) End-to-end detection at 1% FPR against escalation rate `e`, with the cal-frozen operating point
marked; the dashed line is M2 on every input. The knee is sharp — detection saturates once `e` passes
the Stage-1 injection-flagged fraction, because Stage-1 `p_safe` is bimodal OOD. (b) Measured cost
reduction and break-even `e*`; both operating points sit well left of break-even. Panel (b) is drawn
at a uniform 512-token charge from the first latency pass (28.4% / 27.3%); the re-measured
eight-length curve gives 27.9% / 26.4% and the deployment-relevant length-weighted accounting is in
§3.5 below.*

| Stage 1 | θ_safe (frozen on cal) | escalation `e` | cascade TPR | FPR | leaked attacks | gap vs M2 (95% CI) | verdict |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | 0.005 | **0.516** | 0.401 | 0.98% | 639 (9.0%) | [−0.0007, +0.0056] | **matches** M2 |
| Qwen-2.5-1.5B | 0.005 | **0.397** | 0.415 | 0.99% | 460 (6.5%) | [+0.0040, +0.0225] | **exceeds** M2 |

So ~48% (Llama) / ~60% (Qwen) of requests are resolved by the small model with no measurable
detection loss. The cost of Qwen's +1–2 pt gain is 460 **permanently leaked** attacks — auto-passed,
and therefore uncatchable at any Stage-2 quality.

This positive result is only visible with the **real per-row Stage 2**. An earlier sweep that modelled
Stage 2 as a fixed `(DR, FPR)` pair concluded the cascade could not save cost
(`FINDINGS_cascade_cost_frontier.md`, superseded) — it could not see the one effect that matters,
because Stage 2's threshold adapts to the escalated subpopulation.

### 3.4 The detection ceiling is one channel, not a general weakness

![Detection rate by injection channel](docs/figures/channel_dr_double.png)

*DR@1%FPR decomposed by the channel an injection arrives through (2,830 direct / 2,819 document /
1,464 tool attacks).*

| channel (source) | attacks | cascade DR | leaked (auto-passed) |
|---|---|---|---|
| **direct** (struq_synthetic) | 2,830 | **0.92–0.93** | 3 |
| tool (agentdojo) | 1,464 | 0.08–0.10 | 58–66 |
| **document** (openpromptinjection) | 2,819 | **0.05–0.07** | 399–570 |

The aggregate ≈0.40 is a *composition*: direct prompt injection is nearly solved, and
indirect/document-embedded injection is almost entirely missed — by both stages. Leaks are not
diffuse; they concentrate in the document channel. Re-scoring every system with the document channel
removed (22,928 rows) confirms it:

| system | DR@1%FPR, full eval | document channel removed | Δ |
|---|---|---|---|
| M2 (Stage 2) | 0.402 | **0.631** | +0.230 |
| Cascade Llama→M2 | 0.403 (e = 0.516) | **0.633** (e = 0.481) | +0.230 |
| Cascade Qwen→M2 | 0.418 (e = 0.397) | **0.648** (e = 0.340) | +0.230 |

The cascade still matches M2 on the cleaned set, and because the removed attacks were the ones being
escalated, `e` *drops* — the cost saving widens. **Confound to state:** channel and source are 1:1 in
this benchmark (direct = struq, tool = agentdojo, document = openpromptinjection), so channel effects
are also source effects; struq-style direct attacks may resemble the training construction while
openpromptinjection is genuinely unseen. The defensible one-liner is *the cascade cheaply resolves
direct injection and neither stage handles indirect injection* (E6a, E6c, E10).

### 3.5 Cost — measured, and length-dependent

Measured on A10G, batch = 1, median of 40 reps. `e` is the escalation rate that Stage-1 precision
actually produces (bf16 and NF4 scores differ, so the same frozen θ_safe routes a different fraction).

**Uniform 512-token charge** — every request billed at 512 tokens:

| Stage-1 regime | Stage 1 | k₁ (ms) | k₂ (ms) | `e` | break-even `e*` | **cost reduction** |
|---|---|---|---|---|---|---|
| NF4 | Llama-1B | 111.4 | 543.2 | 0.516 | 0.795 | **+27.9%** |
| NF4 | Qwen-1.5B | 184.1 | 543.2 | 0.397 | 0.661 | **+26.4%** |
| bf16 | Llama-1B | 98.5 | 543.4 | 0.444 | 0.819 | **+37.5%** |
| bf16 | Qwen-1.5B | 157.6 | 543.4 | 0.377 | 0.710 | **+33.3%** |

**Length-weighted** — each request charged at its own rendered length, which is the deployment-relevant
figure. This benchmark's traffic is short: median 144 rendered prompt tokens, only 10% of requests
reaching 512.

| Stage-1 regime | Stage 1 | mean k₁ (ms) | mean k₂ (ms) | `e` | break-even `e*` | **cost reduction** |
|---|---|---|---|---|---|---|
| NF4 | Llama-1B | 104.0 | 324.2 | 0.516 | 0.679 | **+2.3%** |
| NF4 | Qwen-1.5B | 178.6 | 324.2 | 0.397 | 0.449 | **−2.2%** |
| bf16 | Llama-1B | 75.0 | 323.5 | 0.444 | 0.768 | **+20.0%** |
| bf16 | Qwen-1.5B | 124.3 | 323.5 | 0.377 | 0.616 | **+17.4%** |

**Stated honestly: the cascade cuts inference cost 26–37% at a uniform 512-token charge, but −2% to
+20% on this benchmark's actual length mix — and the whole saving there comes from running Stage 1 in
bf16.** At $1.20/GPU-hr and length-weighted accounting that is $107.84 → **$86.28** per million
requests (Llama bf16) or **$89.12** (Qwen bf16); an NF4 Stage 1 is a wash in both directions, because
a 179 ms NF4 router costs about as much as the share of the Stage-2 call it avoids once requests are
this short. Two effects put the measured band at the low end of the 28–45% FLOP-proxy estimate: at
1–1.5B scale fixed dequant and kernel-launch overhead make the small model's wall-clock ratio worse
than its parameter ratio (Qwen measured r = 0.34 vs proxy 0.21), and that overhead does not shrink
with the input. Peak VRAM: Llama 3.71 GB (NF4) / 5.16 GB (bf16), Qwen 4.35 / 6.29, M2 7.52 (NF4).

### 3.6 The operating point is an oracle

![Calibration and threshold non-transfer](docs/figures/calibration_double.png)

*Reliability on the calibration split versus OOD for M2 and Stage 1, and the threshold-transfer
panel. All detectors are near-perfectly calibrated in-distribution (Qwen ECE 0.0003) and badly
miscalibrated OOD (ECE 0.259); a temperature fit on the saturated calibration split does not transfer.
Worse, freezing Stage 2's 1%-FPR **score threshold** on cal and applying it to eval yields
**57.9% FPR** against a 1% target.*

Consequence, stated plainly: **every DR@1%FPR figure in this repository — cascade and baselines alike
— relies on eval-set threshold calibration, an oracle operating point.** Deployment would need either
an OOD-representative calibration set or an OOD-robust thresholding rule. This is a first-order open
problem, not a footnote (E6e).

---

## 4. Repository layout

```
src/
  data/          schema, taxonomy, channel rendering, split builder, synthetic negatives,
                 dedup, and per-source loaders (bipia, agentdojo, injecagent, hackaprompt,
                 tensortrust) + eval_sources (opi_document, agentdojo_tool)
  models/        stage1.py (QLoRA scoring: 2-forward constrained decode), prompt_template.py
  calibration/   temperature_scaling.py, threshold_search.py
  pipeline/      routing.py — the one-sided routing rule
  evaluation/    metrics.py (detection_rate_at_fpr, bootstrap CIs), cost.py, per_channel.py
  utils/         io, logging, seed
scripts/         one entry point per experiment (see §5)
configs/         YAML for data, models, training, evaluation, W&B
notebooks/       data exploration, model selection, Stage-2 fine-tune, figures
tests/           pytest suite
docs/figures/    the result figures embedded above (tracked)
data/, results/  gitignored outputs
```

The end-to-end cascade is assembled in `scripts/eval_cascade.py` (it replays both stages' saved
per-row logits, so it needs no GPU) rather than in `src/pipeline/cascade.py`, which is an unused
stub — as are `src/models/{stage2,promptguard}.py`, `src/evaluation/results_formatter.py` and
`src/utils/wandb_utils.py`. Stage-2 training ran in `notebooks/03_stage2_finetune_kaggle.ipynb`.

Design rationale: `../cascade-pid-structure.md`. Data contract: `SCHEMA.md`.
Dataset datasheet: `datasheet.md` (note: describes the frozen v1 3,673-row corpus; the reported
results use the later `train_proposal` / `eval_proposal` corpora in §2.3).

## 5. Reproducing

```bash
pip install -e .
cp .env.example .env      # WANDB_API_KEY, HF_TOKEN

make all                  # full pipeline
# or step by step:
make data                 # build + audit + freeze the channel-stratified corpora
make train MODEL=qwen2.5-1.5b
make calibrate MODEL=qwen2.5-1.5b
make baselines            # encoder classifiers + DataSentinel-7B
make cascade              # end-to-end cascade evaluation
make sweep                # detection vs escalation frontier
make evaluate
```

| # | script | purpose |
|---|---|---|
| 0 | `download_data.sh`, `download_hf.py` | clone BIPIA / InjecAgent / Open-Prompt-Injection / StruQ from GitHub; HuggingFace sources (AgentDojo, HackAPrompt, TensorTrust, alpaca, lmsys, dolly, …) stream on demand |
| 1 | `build_dataset.py` | build the channel-stratified pool and splits per `SCHEMA.md` |
| 1b | `audit_dataset.py`, `freeze_dataset.py`, `cross_eval_dedup.py` | balance/leakage audit, freeze, cross-corpus dedup |
| 2 | `train_stage1.py` | QLoRA fine-tune the Stage-1 candidates |
| 2b | `build_stage2_trainset.py` | Stage-2 corpus with long-benign augmentation (kills the length shortcut) |
| 3 | `calibrate.py` | temperature scaling + threshold search (diagnostic) |
| 4 | `run_baselines.py`, `recheck_datasentinel_fps.py` | PromptGuard 2 / ProtectAI / DataSentinel-7B standalone |
| 5 | `eval_cascade.py` | end-to-end cascade, bootstrap CIs — the headline |
| 6 | `sweep_theta_safe.py`, `sweep_thresholds.py` | routing-threshold and detection/escalation frontiers |
| 7 | `benchmark_latency_curve.py`, `analyze_length_weighted_cost.py` | measured k₁/k₂ across eight prompt lengths; length-weighted cost |
| 8 | `cascade_ablations.py`, `eval_cascade_twosided.py`, `recheck_contamination.py`, `analyze_failure_overlap.py`, `run_precision_ablation.sh` | ablations E6, E8, E12, E13 |
| — | `make_figures.py` | regenerate every figure + `figure_data.json` (never hand-edit outputs) |
| — | `inspect_samples.py` | payload-safe dataset inspection |

Regenerate figures:

```bash
PYTHONPATH=. python scripts/make_figures.py --out results/figures/thesis
```

W&B project `cascade-pid`; run groups `data-prep`, `model-selection`, `calibration`, `baselines`,
`cascade-eval`, `threshold-sweep`.

## 6. Findings documents

`EXPERIMENTS.md` is the plan-of-record — it states what was run, what was not, and where each
artifact lives. Written-up conclusions:

| document | covers |
|---|---|
| `FINDINGS_e2_stage1_selection.md` | Stage-1 candidate training and selection (E2) |
| `FINDINGS_E5_E7_cascade_headline.md` | **the headline** — cascade, calibration, cost (E3, E5, E6, E7) |
| `FINDINGS_E8_twosided_deny.md` | whether a Stage-1 `DENY` branch helps (E8) |
| `FINDINGS_E9_leak_anatomy.md` | anatomy of the attacks Stage 1 auto-passes (E9) |
| `FINDINGS_E10_seen_vs_unseen_document.md` | seen- vs unseen-corpus generalization (E10) |
| `FINDINGS_E11_stage1_score_sensitivity.md` | cascade sensitivity to Stage-1 score perturbation (E11) |
| `FINDINGS_E12_precision_ablation.md` | NF4 vs bf16 Stage 1 — detection *and* cost (E12) |
| `FINDINGS_E13_failure_overlap.md` | do the two stages fail on the same attacks (E13) |
| `FINDINGS_cost_vs_length.md`, `FINDINGS_short_input_cost.md` | cost as a function of input length |
| `FINDINGS_cascade_cost_frontier.md` | the earlier **superseded** modelled-Stage-2 analysis |
| `RECONCILE_thesis_main6_vs_E5E7.md` | reconciliation of the thesis draft against these results |

## 7. Limitations

Quote these with any number above.

1. **Oracle operating point.** Every DR@1%FPR uses an eval-set-calibrated threshold; the cal-frozen
   threshold gives 57.9% FPR on eval (§3.6). Not deployment-realisable as stated.
2. **Detection ceiling.** M2 itself reaches only 0.402 at 1% FPR on this benchmark. The cascade
   inherits that ceiling; it does not raise it.
3. **Indirect injection is unsolved.** 0.05 detection in the document channel, by both stages.
4. **Leaked-attack floor.** 6.5% (Qwen) / 9.0% (Llama) of attacks are auto-passed by Stage 1 and
   uncatchable regardless of Stage-2 quality; they concentrate in the document channel.
5. **Channel–source confound.** Channel and source are 1:1, so the 0.92 vs 0.05 split is plausibly
   in part a seen- vs unseen-distribution effect.
6. **Cost regime.** k₁/k₂ are measured but single-stream (batch = 1) on one GPU class (A10G).
   Batching amortises exactly the fixed overhead that makes NF4 expensive at short lengths, so it
   should move the NF4 arms up — that regime is not measured. The two NF4 length-weighted results
   sit within ±2.3% of zero, inside plausible session-to-session drift; treat "NF4 is roughly
   cost-neutral on this traffic" as the claim, not the sign.
7. **Stage-1 adapters are not full-corpus.** Every Stage-1 number comes from a 12,000-row, 1-epoch,
   seq-1024 QLoRA adapter (Granite: 9,000 rows, a Kaggle 12h wall-clock limit). The planned
   full-corpus re-train was never run. Stage 2 *was* trained on the full 50,349-row corpus.
8. **Stage-1 selection is test-set selection.** Validation saturates for all three candidates
   (AUROC 1.000), so selection rests on the OOD eval set — and the winner changes with the metric
   (eval AUROC → Qwen; eval DR@1%FPR → Llama; saturated-val ECE tie-break → Granite). Both Qwen and
   Llama are therefore carried end-to-end and reported as a two-arm comparison rather than resolved
   by fiat.
9. **Positional join.** Stage-1 logit files carry `{logp_benign, logp_injection, p_safe}` with no
   `id`; alignment to the split is positional, corroborated by per-source label purity. Future
   scoring should emit `id`.
10. **Adapter drift.** The Qwen eval was scored with a box copy that drifts 0.5% mean / 8.6% max in
    `p_safe` from the repo copy; ranking metrics move < 1.5 pt. Llama is bit-exact to its Kaggle logits.
11. **No end-to-end deployment study.** The RAG / course-assistant case study measuring
    attack-execution rate and over-blocking in a live pipeline was not run.

## 8. Citation

```bibtex
@mastersthesis{le2026cascadepid,
  title  = {Cascaded Prompt Injection Detection Using Small Language Models},
  author = {Le, Huy},
  school = {RMIT University},
  year   = {2026}
}
```
