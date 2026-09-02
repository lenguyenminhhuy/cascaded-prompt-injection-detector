# Short-length latency and the length-weighted cost result

**Experiment:** Stage-1/Stage-2 latency at 32 and 64 tokens, plus verification of the
positional logit join and correction of the determinism caveat
**Hardware:** NVIDIA A10G, g5 class (`ec2-100-23-178-248`), measured 2026-08-31
**Protocol:** batch=1, median of 40 reps after 8 warm-ups. Stage 2 always nf4.
**Artifacts:** `results/analysis/latency_curve_nf4.json`,
`results/analysis/latency_curve_bf16stage1.json`,
`results/analysis/cost_length_weighted_short_regime.json`

## 1. There is no sub-86-token regime

A 32- and a 64-token target render to the **same** actual prompt length, because the
chat template plus instruction wrapper floors every request:

| model | floor (actual prompt tokens) |
|---|---|
| llama3.2-1b | 87 |
| qwen2.5-1.5b | 86 |
| mistral-7b-v0.1 | 83 |

So "two thirds of requests fall below the shortest length timed" is better stated as
"two thirds of requests sit at the template floor". The floor is now measured.

## 2. Stage 1's flat-hold was right; Stage 2's was not

Stage-1 latency is flat from the floor to 128 tokens (+0.4%). Stage 2 is not:
mistral-7b costs **184.0 ms at 83 tokens vs 223.4 ms at 127** (-17.6%). The previous
flat-hold therefore **overcharged k2 by ~18% on the two thirds of requests at the floor**.

## 3. Correcting the floor moves the result against the cascade

Three arms, so the structural effect is separated from measurement noise:

- **A** — pre-existing curve, 128/512/1024/2048 anchors (reproduces the published numbers)
- **B** — new curve, same anchors → isolates session-to-session drift
- **C** — new curve plus the ~86-token anchor → the corrected number

Length-weighted reduction on the full 25,747-row eval:

| Stage-1 regime | model | A (published) | B (drift) | C (corrected) | drift | anchor |
|---|---|---|---|---|---|---|
| nf4 | qwen2.5-1.5b | -4.0% | -5.4% | **-12.5%** | -1.4 pt | **-7.1 pt** |
| nf4 | llama3.2-1b | +4.4% | +3.6% | **-2.5%** | -0.8 pt | **-6.2 pt** |
| bf16 | qwen2.5-1.5b | +16.3% | +15.9% | **+11.0%** | -0.4 pt | **-4.9 pt** |
| bf16 | llama3.2-1b | +21.5% | +21.8% | **+16.7%** | +0.3 pt | **-5.1 pt** |

The anchor effect (-4.9 to -7.1 pt) is roughly six times the session drift (±1 pt).

**Why it moves against the cascade.** The baseline pays k2 on *every* input, so
cheapening k2 at the floor cheapens the baseline faster than it cheapens the cascade,
whose cost is dominated by k1 running on all 25,747 inputs. The naive expectation
(lower k2 helps the cascade) is backwards.

## 4. The sign depends on a Stage-1 quantization choice

nf4 Stage-1 is cost-negative on length-weighted traffic; bf16 Stage-1 is clearly positive.
The cause is that 4-bit dequantization overhead dominates at batch=1 for a 1-2B model:

| model | k1 @512, nf4 | k1 @512, bf16 |
|---|---|---|
| llama3.2-1b | 109.9 ms | 96.1 ms |
| qwen2.5-1.5b | 185.8 ms | 138.6 ms |

A 1-2B Stage 1 does not need 4-bit to fit on a 23 GB card, so **bf16 is the correct
deployment regime** and 4-bit was a training-time convenience. The published
length-weighted figures quote the nf4 arm, i.e. the worse of the two.

**The uniform-512 headline is unaffected**: llama 28.2% / qwen 26.2% (nf4),
llama 30.8% / qwen 34.8% (bf16), spanning the reported 27-35% band.

## 5. Cross-session variance is length-dependent

Re-measuring the shared anchors in a fresh session: **<=0.3% at 1024-2048 tokens, but
1.8-3.7% at 64-512**. Fixed overhead dominates at short lengths and is more sensitive to
clock and thermal state. The paper should quote the larger figure; it remains small
against the -6 pt anchor effect.

## 6. The positional join is verified, and the provenance caveat is wrong

`scripts/score_split.py` now emits `id` and refuses a partially keyed dump. A 300-row
random probe (seed 0, eval positions 37-25,724) re-scored with ids and compared against
the rows the positional join selects:

```
id-position agreement: 300/300
p_safe exact=300  mismatched=0  max |delta| = 0.000e+00
VERIFIED: positional join was correct
```

Separately, the report's claim that Qwen's scores "come from a checkpoint copy whose mean
scores differ by 0.5% from the repository copy" does not match the artifacts:

| comparison | mean rel. diff | max abs delta | exact rows |
|---|---|---|---|
| box vs laptop adapters (all 3 models) | - | - | bit-identical (sha256) |
| `stage1/qwen` vs Kaggle eval logits | 0.000% | 0.00000 | 25,747 / 25,747 |
| `stage1/qwen` vs `stage1_prec/nf4/qwen` | 0.000% | 0.00000 | 25,747 / 25,747 |
| nf4 vs bf16, eval | **5.159%** | 0.97180 | 1,066 / 25,747 |
| nf4 vs bf16, cal | 0.023% | 0.52909 | 401 / 5,722 |

Provenance is clean and nf4 scoring is bit-reproducible across sessions. The only real
difference is the **quantization regime**, and on eval it is ~10x larger than the stated
0.5%. `FINDINGS_E11` already establishes that DR is insensitive to score drift of this
size (<1.5 pt), which is the load-bearing half.

## 7. Consequences for `thesis_report.tex`

1. `sec:exp-uncertainty` - the hedge "the sign of Qwen's -4.0% is not firmly established
   and neither is Llama's +4.4%" resolves, against the cascade for nf4 and in favour for
   bf16. Replace with the measured pair and name the regime.
2. `sec:exp-uncertainty` - drop the positional-join caveat (now verified) and rewrite the
   drift sentence as a quantization-regime difference, not checkpoint provenance.
3. `sec:exp-limitations` - "the shortest length timed is 128 tokens, below which two thirds
   of inputs fall" is superseded: the floor is 83-87 tokens and is now measured.
4. Consider reporting the bf16 Stage-1 arm as the deployment configuration.

## 8. Not addressed

Batched throughput remains unmeasured; everything here is batch=1.
