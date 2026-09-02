# E5 + E7 — Cascade end-to-end evaluation with the REAL Stage-2 (headline result)

**Status:** complete, independently reviewed (code-correctness + methodology).
**Script:** `scripts/eval_cascade.py` → `results/analysis/cascade_{llama3,qwen2}/cascade_summary.json`
**Date:** 2026-08-19; cost section re-measured 2026-09-01/02. **No GPU** for the detection
results (post-hoc on saved logits); latency measured on an A10G box. **Payload-safe** (labels + logits only).

---

## 1. What is new vs the earlier (negative) cost-frontier finding

`FINDINGS_cascade_cost_frontier.md` swept routing thresholds against a **modeled** Stage-2
(a fixed `(DR, FPR)` pair, `--stage2-model perfect|param`) on the small 253/366/523 test
splits, and concluded the cascade could not save cost. That analysis could not see the
one effect that actually matters, because it modeled Stage-2 as a constant.

This run uses the **actual per-row Stage-2 (Mistral-7B QLoRA) predictions** on the full
25,747-row OOD eval set, so Stage-2's threshold can adapt to the *escalated subpopulation*.
That flips the result from negative to **positive** (see §4).

---

## 2. Data & single-stage baselines (DR@1% FPR, threshold set on eval benign)

Eval = 25,747 rows (7,113 attacks / 18,634 benign), out-of-distribution.

| detector | params | AUROC | **DR@1%FPR** | eval ECE |
|---|---|---|---|---|
| Stage-1 llama3.2-1b | 1.24B | 0.882 | 0.189 | 0.35 |
| Stage-1 qwen2.5-1.5b | 1.54B | 0.929 | 0.171 | 0.26 |
| Stage-2 mistral-7b | 7.24B | 0.934 | **0.402** | 0.20 |

**Sobering context:** even the strong 7B Stage-2 catches only **40%** of attacks at 1% FPR on
this hard OOD eval. The cascade's detection ceiling is therefore ~0.40, not ~1.0. This is a
property of the eval set / detectors, independent of the cascade.

## 3. E3 — Calibration diagnostic (temperature scaling)

T fit on the in-distribution cal split; ECE before→after.

| detector | T (cal) | ECE cal raw→temp | ECE **eval** raw→temp |
|---|---|---|---|
| llama3.2-1b | 0.86 | 0.001→0.001 | 0.347→0.349 |
| qwen2.5-1.5b | 1.23 | 0.000→0.000 | 0.259→0.258 |
| mistral-7b | 0.59 | 0.184→0.154 | 0.204→0.211 |

**Finding:** all detectors are ~perfectly calibrated in-distribution but badly miscalibrated
OOD, and a temperature fit on the (saturated) cal split **does not transfer** — it barely
moves eval ECE. DR@fixed-FPR and the routing frontier are rank-based, hence invariant to
monotone T, so calibration is reported as a diagnostic only and the frontier is computed on
raw scores. (The non-transfer is itself the E6e "calibration/threshold OOD transfer" story.)

## 4. E5 — Cascade headline (one-sided obvious-benign filter)

**Design (an empirical result, not an assumption):** the cascade is one-sided — Stage-1
auto-passes rows with `p_safe ≥ θ_safe` (predict benign, skip Stage-2); everything else
escalates to Stage-2. Two-sided auto-block is **never budget-positive** here: an auto-block of
benign spends the global 1%-FPR budget that Stage-2 would spend more accurately, and Stage-1
is too miscalibrated OOD to auto-block safely. So Stage-1 functions purely as a cheap
**benign pre-filter**; all cost savings come from passing benign, all end-to-end false
negatives from leaked auto-passed attacks.

**Operating point θ_safe is selected on cal and FROZEN, then reported on eval** (no test-set
argmax). The Stage-2 threshold is set on eval benign to hit overall 1% FPR — the *same*
convention as the single-stage baselines, so the comparison is apples-to-apples (each sets
exactly one eval threshold). 1000× paired bootstrap on eval.

| Stage-1 | θ_safe (cal) | eval **e** (escalation) | eval cascade **TPR** | eval FPR | leaked attacks | gap vs Stage-2 (95% CI) | verdict |
|---|---|---|---|---|---|---|---|
| llama3.2-1b | 0.005 | **0.516** | 0.401 | 0.98% | 639 (9.0%) | [−0.0007, +0.0056] | **matches** Stage-2 |
| qwen2.5-1.5b | 0.005 | **0.397** | 0.415 | 0.99% | 460 (6.5%) | [+0.0040, +0.0225] | **exceeds** Stage-2 |

**Reading it:**
- **llama:** escalating only **51.6%** of traffic to the 7B model preserves Stage-2's full
  detection at 1% FPR (gap CI straddles 0 → statistically indistinguishable). ~48% of requests
  are resolved by the 1B model alone with no measurable detection loss.
- **qwen:** escalating only **39.7%** *exceeds* Stage-2-only detection (gap CI entirely > 0).
  This is real, not a selection artifact (θ_safe frozen on cal, effect survives bootstrap):
  auto-passing easy benign concentrates the fixed 1% FP budget onto a smaller, harder
  escalated-benign pool, letting Stage-2 run a looser threshold (θ2 0.194 vs 0.207 standalone)
  and catch more of the escalated attacks. **Caveat:** this buys +1–2 pt TPR at the cost of
  460 permanently-leaked attacks (auto-passed, uncatchable at any Stage-2 quality).

**Eval-optimal upper bounds** (θ_safe argmax'd on eval — optimistic, not deployment-realizable,
reported for completeness): llama TPR 0.402 @ e=0.603; qwen TPR 0.415 @ e=0.397.

## 5. E7 — Cost

Cost model `C(e)=k1+e·k2`, break-even `e<1−k1/k2` (`src/evaluation/cost.py`).

**k1/k2 are MEASURED.** Re-measured 2026-09-01 with `scripts/benchmark_latency_curve.py` →
`results/analysis/latency_curve_full_{nf4,bf16}.json`, which times eight prompt lengths
(64–2048 tokens) instead of the single 512-token point used in the first pass
(`latency_measured{,_bf16stage1}.json`, 2026-08-20 — **superseded**; the numbers below replace
it). Regime: batch=1 single-stream, NVIDIA A10G, median of 40 reps (8 warmup), model load +
warmup excluded. Each per-input call is the same 2-forward constrained-decode scoring used to
produce the logits, so both stages are timed in the identical regime.

### 5a. Uniform-512 accounting (every request charged at 512 tokens)

| regime | Stage-1 | k1 (ms) | k2=M2 (ms) | r=k1/k2 | e | break-even e | **reduction @ e** |
|---|---|---|---|---|---|---|---|
| NF4 both (parity w/ scored logits) | llama3.2-1b | 111.4 | 543.2 | 0.205 | 0.516 | 0.795 | **+27.9%** |
| NF4 both | qwen2.5-1.5b | 184.1 | 543.2 | 0.339 | 0.397 | 0.661 | **+26.4%** |
| bf16 S1 / NF4 S2 (deploy lever) | llama3.2-1b | 98.5 | 543.4 | 0.181 | 0.444 | 0.819 | **+37.5%** |
| bf16 S1 / NF4 S2 | qwen2.5-1.5b | 157.6 | 543.4 | 0.290 | 0.377 | 0.710 | **+33.3%** |

Each row uses the escalation rate that precision actually produces. NF4 and bf16 Stage-1 scores
differ by ~5.2% on eval, so the same frozen `θ_safe=0.005` routes a different fraction; an earlier
version of this table reused the NF4 `e` (0.516 / 0.397) in the bf16 rows, which understated bf16.
The bf16 rows are no longer a cost-only lever: **bf16 Stage-1 was scored for detection too**, and
it changes the detection story as well as the cost one (qwen solo DR 0.171 → 0.364, cascade DR
0.4218; llama leaks 639 → 962 attacks). See `FINDINGS_E12_precision_ablation.md` — §2/§3/§4 above
report the NF4 arms.

$ / GPU-seconds at $1.20/GPU-hr: M2-every-input = **$181.1/1M**; llama cascade **$130.6/1M** (NF4)
and **$113.2/1M** (bf16); qwen cascade **$133.3/1M** (NF4) and **$120.8/1M** (bf16).

### 5b. Length-weighted accounting (each request charged at its own length)

The eval's own traffic is short: the median request renders to 144 prompt tokens against a 512-token
charge, and only 10% of requests reach 512. Charging every request 512 tokens therefore flatters the
cascade, because Stage-2's cost grows steeply with length while Stage-1's is nearly flat (fixed
dequant + launch overhead dominates below ~256 tokens). Re-running the same cost model with
per-request lengths (`scripts/analyze_length_weighted_cost.py` →
`results/analysis/cost_lwfull_*.json`, 8-anchor interpolation, regenerated 2026-09-02):

| regime | Stage-1 | mean k1 (ms) | mean k2 (ms) | r_eff | e | break-even e | **reduction @ e** |
|---|---|---|---|---|---|---|---|
| NF4 both | llama3.2-1b | 104.0 | 324.2 | 0.321 | 0.516 | 0.679 | **+2.3%** |
| NF4 both | qwen2.5-1.5b | 178.6 | 324.2 | 0.551 | 0.397 | 0.449 | **−2.2%** |
| bf16 S1 / NF4 S2 | llama3.2-1b | 75.0 | 323.5 | 0.232 | 0.444 | 0.768 | **+20.0%** |
| bf16 S1 / NF4 S2 | qwen2.5-1.5b | 124.3 | 323.5 | 0.384 | 0.377 | 0.616 | **+17.4%** |

**Measured headline, stated the honest way: the cascade cuts inference cost by 26–37% if every
request is charged at 512 tokens, but by −2% to +20% on the eval's actual length mix — and the
whole saving there comes from running Stage-1 in bf16.** With bf16 Stage-1 the cascade saves
17–20% ($107.84 → $86.28 per 1M for llama, → $89.12 for qwen); with NF4 Stage-1 it is a wash in
both directions (+2.3% llama, −2.2% qwen), because a 179 ms NF4 router costs about as much as the
share of the Stage-2 call it avoids once requests are this short.

Lengths here are **rendered prompt** tokens — the request put through `format_prompt` and tokenized
per stage with `add_special_tokens=False` — which is the same quantity the latency curve records as
`actual_prompt_tokens`. An earlier version of this analysis counted the raw payload only, pricing
every request ~74 tokens too far left on the curve (the template's own length); because Stage-2's
curve is steeper, that understated k2 more than k1 and so understated the reduction, by 2.1 points
(bf16 llama) to 7.2 points (NF4 qwen). The fix also disposes of the old flat-hold caveat: the curve's floor
*is* the empty-prompt length, so no request can fall below it, and only 0.06% run past the
2049-token top anchor. Essentially every request is now priced on measured ground.

Two effects explain why the measured band sits at the low end of the earlier ~28–45%
FLOP/param-proxy estimate. (1) At 1–1.5B scale, fixed dequant and kernel-launch overhead make the
small model's wall-clock ratio worse than its parameter ratio (qwen measured r=0.34 vs proxy 0.21).
(2) That overhead does not shrink with the input, so on short requests the router is a large
fraction of an already-cheap Stage-2 call.

Caveats: single-stream (batch=1) on one GPU class (A10G). Batching amortises exactly the fixed
overhead that makes NF4 expensive at short lengths, so it should move the NF4 arms up; that regime
is not measured. The two NF4 results sit within ±2.3% of zero, which is inside the range
session-to-session latency drift can produce (§5 of `FINDINGS_cost_vs_length.md` puts ratio drift
under 1.5%, but that is one re-measurement, not a distribution) — treat "NF4 is roughly cost-neutral
on this traffic" as the claim, not the sign. Peak VRAM: llama 3.71 GB (NF4) / 5.16 GB (bf16),
qwen 4.35 / 6.29, M2 7.52 (NF4).

## 6. Limitations (must accompany any quotation of these numbers)

1. **Cost regime** — k1/k2 are measured (see §5), but single-stream (batch=1) on one GPU class
   (A10G); a batched-throughput serving regime, under which r=k1/k2 and the reduction can shift,
   is not measured. The headline reduction also depends on how requests are charged: +26–37% at a
   uniform 512 tokens, −2% to +20% on the eval's real length mix (§5b). Quote the length-weighted
   figure as the deployment-relevant one, and name the Stage-1 precision with it.
2. **Detection ceiling is low** — Stage-2 itself only reaches DR 0.40 @1%FPR on this OOD eval;
   the cascade matches/inherits that, it does not fix it.
3. **Leaked-attack floor** — 6.5% (qwen) / 9.0% (llama) of attacks are auto-passed and
   uncatchable regardless of Stage-2. The per-source breakdown (E6a, §6b) shows these leaks are
   not diffuse: they concentrate almost entirely in the document channel.
4. **Row-order join** — logit files carry only `{logp_benign, logp_injection, p_safe}` (no id);
   alignment to the split is positional, corroborated externally by per-source label purity
   (alpaca pos=0, bipia neg=0 in `score_stage1_logits.py`). Future scoring should emit `id`.
5. **Stage-2 threshold** is eval-set-calibrated to 1% FPR (standard, and matched by the
   baselines); whether it transfers from cal is the separate E6e robustness check.
6. **qwen adapter provenance** — the eval was scored with a box copy that drifts 0.5% mean /
   8.6% max in p_safe from the repo copy; ranking metrics move <1.5 pt (immaterial). llama is
   bit-exact to its Kaggle logits.

## 6b. E6 ablations (post-hoc, `scripts/cascade_ablations.py`)

**E6a — leaks are channel-structured, not diffuse (major refinement of the headline).**
At the cal-frozen operating point, detection decomposes almost entirely by channel:

| channel (source) | attacks | detection rate | leaked (auto-pass) |
|---|---|---|---|
| direct (struq_synthetic) | 2,830 | **0.92–0.93** | 3 |
| tool (agentdojo) | 1,464 | 0.08–0.10 | 58–66 |
| document (openpromptinjection) | 2,819 | **0.05–0.07** | 399–570 |

So the aggregate DR≈0.40 is a *composition* of "**direct** prompt injection is nearly solved
(0.92)" and "**indirect / document-embedded** injection is almost entirely missed (0.05)" —
by **both** stages, not just Stage-1. The leaked (auto-passed) attacks concentrate in the
document channel. **Confound to state:** in this eval, channel and source are 1:1
(direct=struq, tool=agentdojo, document=openpromptinjection), so channel effects are source
effects. `struq`-style direct attacks may resemble training data while `openpromptinjection`
is genuinely OOD — i.e. the 0.92 vs 0.05 split is plausibly a seen-vs-unseen-distribution
effect (relates to E6c contamination check). The honest one-line result is: *the cascade
cheaply resolves direct injection and neither stage handles indirect injection.*

**E6c — the ~0.40 ceiling IS the document channel (contamination re-score).**
`scripts/recheck_contamination.py` → `results/analysis/contamination_recheck.json`. Re-scoring
every system on eval **minus source=openpromptinjection** (22,928 of 25,747 rows; the document
channel removed) raises DR@1%FPR sharply and uniformly:

| system | DR@1%FPR full | −OPI | Δ |
|---|---|---|---|
| M2 (Stage-2) | 0.402 | **0.631** | +0.230 |
| llama3.2-1b | 0.189 | 0.291 | +0.102 |
| qwen2.5-1.5b | 0.171 | 0.283 | +0.112 |
| cascade llama→M2 (TPR) | 0.404 (e=0.516) | **0.633** (e=0.481) | +0.230 |
| cascade qwen→M2 (TPR) | 0.418 (e=0.397) | **0.648** (e=0.340) | +0.230 |

Reading: the low 0.40 aggregate is **almost entirely the single unsolved document channel**, not
a general weakness — remove it and the guard/cascade catch ~63–65% at 1% FPR. The cascade still
matches M2-alone on the cleaned set (llama 0.633 vs 0.631; qwen 0.648 exceeds), and since the
removed document attacks were the ones being escalated, **e drops** (0.52→0.48, 0.40→0.34) — the
cost saving *widens*. Caveat: this does NOT clear the direct channel of a seen-distribution
advantage (struq may resemble the training construction); it localises the headline limitation to
indirect/document-embedded injection. Leaked attacks on the kept (non-document) set are small
(69 llama / 61 qwen), confirming leaks concentrate in the document channel.

**E6e — the 1% FPR operating point does NOT transfer OOD.** Freezing Stage-2's 1%-FPR score
threshold on cal and applying it to eval yields **eval FPR = 57.9%** (target 1%). The benign
score distribution shifts so much under OOD that the cal-calibrated threshold is meaningless
on eval. Consequence: every "DR@1% FPR" number here (cascade **and** single-stage baselines)
relies on **eval-set threshold calibration** — an oracle operating point. Deployment would
need either an OOD-representative calibration set or an OOD-robust thresholding method; this is
a first-order open problem, not a footnote.

Frontier figure: `results/figures/cascade_frontier.png` (TPR vs escalation at 1% FPR, both
Stage-1 options). The knee is sharp — TPR saturates at ~0.40 once e passes the Stage-1
injection-flagged fraction, because Stage-1 p_safe is bimodal (overconfident) OOD.

## 7. Core claim (survives review, weakened wording)

> A cheap Stage-1 obvious-benign filter escalates only **~40% (qwen) / ~52% (llama)** of traffic
> to the 7B Stage-2 while **matching** its detection at 1% FPR (qwen modestly exceeds it, gap
> CI > 0). Inference cost falls by a **measured 26–37%** when every request is charged at 512
> tokens, and by **17–20%** on the eval's actual (much shorter) length mix provided Stage-1 runs in
> bf16 — an NF4 Stage-1 is roughly cost-neutral there (+2.3% / −2.2%), best case bf16 llama at
> **+20.0%** (same-hardware batch=1 latency on A10G; §5). This positive result is visible only with
> the **real per-row Stage-2**; the earlier modeled-Stage-2 sweep missed it.
>
> **Three first-order caveats that must travel with this claim:** (1) the shared detection level
> (~0.40) is "direct injection solved (0.92), indirect/document injection unsolved (0.05)" — the
> cascade saves cost on the *easy* channel and neither stage handles the hard one (E6a); (2) the
> "1% FPR" is an **eval-calibrated oracle** operating point — it does not transfer from cal
> (57.9% eval FPR when frozen, E6e); (3) the cost saving is length-dependent and collapses to
> roughly zero on short traffic with an NF4 Stage-1 (§5b), so the operating configuration (bf16
> Stage-1) is part of the claim, not an implementation detail. The cost saving is real in the bf16 configuration; the
> *security* value is bounded by both a low OOD detection ceiling and an unsolved OOD-thresholding
> problem.

## 8. Reviews applied

- **Code review:** fixed NaN-robust FP/TP reconstruction (from `detection_rate_at_fpr`'s own
  surviving counts), added resolvability capture, coerced non-finite θ2 → `None` (strict-JSON
  valid), removed dead code, replaced the vacuous row-order assert with a documented convention.
- **Methodology critique:** froze θ_safe on cal (removed test-set argmax), added 1000× bootstrap
  CIs (decides matches-vs-exceeds), reframed cost as a k1/k2 curve + FLOP proxy (dropped the
  unsupported single "+40–60%"), documented the one-sided-by-optimization finding and the
  leaked-attack floor.
