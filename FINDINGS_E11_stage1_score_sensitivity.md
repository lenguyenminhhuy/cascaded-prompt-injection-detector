# E11 — Stage-1 score-perturbation sensitivity of the cascade

> **Status: the question this experiment was built to answer has since been settled
> directly.** E11 was a *proxy*: bf16 Stage-1 detection had not been scored, so the
> effect of a precision change was simulated by perturbing the NF4 scores. The
> precision ablation (`results/analysis/E12_precision_ablation.json`, "COMPLETE — all 4
> passes done") later scored both precisions for real on one A10G session, and its NF4
> control reproduced the paper's NF4 logits **bit-exactly** (both arms, md5-matched),
> so the bf16-vs-NF4 deltas are precision alone. Measured bf16 numbers now replace the
> simulated ones: qwen `e`=0.3772 / DR 0.4218 / 513 leaked; llama `e`=0.4442 / DR 0.4146
> / 962 leaked. **Read E11 for the sensitivity *mechanism* (below), not for the bf16
> estimates.**

**Original question.** Table 6's bf16 Stage-1 block reported 30.8%/34.8% cost reduction
but (a) detection had never been scored in bf16 and (b) the rows reused the escalation
rate `e` measured under NF4. Does the cascade's reported performance survive a change
in Stage-1 scores of the kind a precision change induces?

**Method.** Perturb the Stage-1 logit gap z = logp_benign − logp_injection on **both** the
calibration and evaluation splits, recompute p_safe = sigmoid(z), re-select theta_safe on
cal by the frozen procedure, and evaluate on eval. Stage-2 (M2) scores are untouched — M2
stays NF4 in both Table 6 blocks. Two perturbation families: zero-mean Gaussian noise
(scale sigma, 20 seeds), which `scripts/sensitivity_stage1_noise.py` implements, and
systematic shift (bias b), which is the more realistic model of quantisation error but was
run outside the committed script (see the reproducibility note at the end).

**Harness validation.** At sigma=0 the script reproduces the paper exactly:
Qwen DR 0.4152 / e 0.3969 / leaked 460; Llama DR 0.4012 / e 0.5161 / leaked 639;
M2-on-every-input 0.4015. Runtime ~1 s (offline replay, no model inference).

## Result 1 — detection is robust

Zero-mean noise, sigma up to 5.0 logits: DR moves <=1.3pp for both models; every
point passes the pre-registered 2pt criterion.

Systematic shift b in [-5,+5]: Llama DR holds within 0.5pp throughout. Qwen holds
within 2pt for |b| <= 1 and drifts to +2.3..+2.7pp at b >= 2 — a violation of the
"within 2 percentage points" criterion, but in the benign direction (higher DR).

*Mechanism.* The routing boundary sits at z = -5.29 (theta_safe = 0.005) while the
score distribution spans -28..+20 with mean |z| ~9-11. Only 2.7% (Qwen) / 1.8%
(Llama) of evaluation rows lie within 0.5 logits of the boundary, so a perturbation
must be large to reroute a meaningful share of traffic.

## Result 2 — the cost reduction is NOT robust (the real finding)

`e` is far more sensitive than DR. Reduction 1-(k1+e*k2)/k2 recomputed at the
measured `e` for each shift:

| config      | b=-5  | b=-1  | b=0 (as reported then) | b=+1  | b=+5  |
|-------------|-------|-------|------------------------|-------|-------|
| Qwen NF4    | 14.2% | 24.6% | **27.3%**              | 30.0% | 36.0% |
| Qwen bf16   | 21.8% | 32.2% | **34.9%**              | 37.6% | 43.6% |
| Llama NF4   | 18.3% | 26.5% | **28.4%**              | 30.2% | 37.1% |
| Llama bf16  | 20.7% | 28.9% | **30.8%**              | 32.6% | 39.5% |

A systematic shift of only +-1 logit puts the Qwen bf16 figure anywhere in
32.2-37.6%; +-5 logits gives 21.8-43.6%. The reported figure was a single point
estimate with no error bar, sitting mid-band.

**Two things to know before quoting this table.** First, every cell was computed with
the 2026-08-20 latency constants (k1 179.5 / 138.3 / 108.9 / 95.8 ms, k2 543.5 ms), which
the 2026-09-01 eight-length re-measurement superseded; at b=0 the current uniform-512
values are 26.4% (qwen NF4), 33.3% (qwen bf16), 27.9% (llama NF4) and 37.5% (llama bf16),
and on length-weighted accounting -2.2% / +17.4% / +2.3% / +20.0%. Second, the bf16 rows
here still carry NF4's `e`; E12 measured the real bf16 `e` as *lower* (0.3772 and 0.4442),
which moves bf16 in the favourable direction. So the levels in this table are stale in
both directions and should not be quoted. **What transfers is the slope:** the cost
reduction moves 1.8 (llama) to 2.7 (qwen) percentage points per logit of systematic
Stage-1 score shift, which is large relative to the differences the thesis draws between
configurations.

## Result 3 — residual FN is the most sensitive metric of all

Auto-passed (leaked) attacks, Qwen: 460 at b=0 -> **718 at b=+1**, i.e. 6.5% ->
10.1% of the 7,113 injections. That is a 3.6pp rise in irrecoverable false
negatives, against a pre-registered safety criterion of "no more than 1 percentage
point above M2 alone". DR simultaneously *improves* (0.4152 -> 0.4215) because
auto-passing more benign traffic concentrates the FPR budget on a smaller escalated
pool. So DR can look stable while the safety criterion silently breaks.

## Implications

1. The contribution's "preserve ... performance" half is better supported than the
   bf16 gap suggested: DR is genuinely insensitive to Stage-1 score drift. E12's
   measured bf16 pass agrees — DR moved +0.7pp (qwen) and +1.3pp (llama), both small.
2. The cost half is the fragile one, and for the reason identified here: `e` is the
   sensitive input. E12 measured it rather than assuming it, which is why the bf16
   figures moved.
3. Residual FN should be reported with a sensitivity band, not as a single count. E12
   bears this out: llama's leaked count rose 639 -> 962 (9.0% -> 13.5% of 7,113) purely
   from the precision change, with DR simultaneously *improving*.

## How it was settled (E12, done)

The plan below was executed on an A10G: Stage-1 scored in both NF4 and bf16 for both
models over eval (25,747) + cal (5,722), with Stage-2's saved logits replayed unchanged.

```
python scripts/score_split.py --config configs/models/qwen2.5-1.5b.yaml \
  --adapter results_kaggle/stage1/qwen2.5-1.5b/adapter \
  --splits data/eval_proposal/eval.jsonl data/train_proposal/cal.jsonl \
  --output-dir results/stage1_prec/bf16/qwen2.5-1.5b --no-4bit   # bf16
# drop --no-4bit for the matched NF4 control on the same hardware
python scripts/eval_cascade.py --stage1-dir results/stage1_prec/bf16/qwen2.5-1.5b \
  --stage2-dir results/stage2/mistral-7b-v0.1 --bootstrap 1000
```

Both precisions ran in the same session, which was the point: the pre-existing NF4 logits
came from Kaggle and the latency constants from an A10G, so the original comparison
confounded precision with hardware. The NF4 control came back byte-identical to the paper
source for both arms and the cascade replay reproduced `e`=0.3969/0.5161, DR=0.4152/0.4012
and leaked=460/639 exactly, so that confound is eliminated. Results:
`results/analysis/E12_precision_ablation.json`.

## Reproducibility gap in this experiment

The Gaussian-noise arm is reproducible: `scripts/sensitivity_stage1_noise.py` writes
`results/analysis/sensitivity_{qwen2.5-1.5b,llama3.2-1b}.json`, which hold the sigma sweep
used in Result 1. The **systematic-shift arm is not** — the committed script has no bias
option, and no artifact stores the per-`b` escalation rates, so the Result 2 and Result 3
tables cannot currently be regenerated from the repo. Adding a `--bias` flag and saving
the shift rows alongside the sigma rows would close this.
