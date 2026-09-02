# Short-length latency and the length-weighted cost result

**Experiment:** Stage-1/Stage-2 latency at 32 and 64 tokens, plus verification of the
positional logit join and correction of the determinism caveat
**Hardware:** NVIDIA A10G, g5 class (`ec2-100-23-178-248`), measured 2026-08-31
**Protocol:** batch=1, median of 40 reps after 8 warm-ups. Stage 2 always nf4.
**Artifacts:** `results/analysis/latency_curve_nf4.json`,
`results/analysis/latency_curve_bf16stage1.json`,
`results/analysis/cost_length_weighted_short_regime.json`

> **Partly superseded 2026-09-02 by `FINDINGS_cost_vs_length.md`**, which re-timed all three
> models at eight lengths in a single session and fixed a unit bug in the cost script. Four things
> here were corrected — the template floor (§1), the length-weighted numbers and this document's
> §3 conclusion, the qwen-bf16 512-token anchor (§4), and the uniform-512 values (§4); each is
> marked in place below. §2, §5, §6 and §7 stand as written.

## 1. There is no sub-86-token regime

A 32- and a 64-token target render to the **same** actual prompt length, because the
chat template plus instruction wrapper floors every request:

| model | floor as measured here | **corrected floor** |
|---|---|---|
| llama3.2-1b | 87 | **75** |
| qwen2.5-1.5b | 86 | **74** |
| mistral-7b-v0.1 | 83 | **71** |

⚠️ **The floor numbers in the middle column are too high.** This session's 32-token target still
carried a few words of padding, so it timed a short *padded* prompt rather than an empty one.
Rendering an empty input through `format_prompt` gives 75 / 74 / 71 tokens, and the 2026-09-02
session's target-64 row lands exactly there, so the true floor is now timed directly
(`FINDINGS_cost_vs_length.md` §2). The structural point of this section is unaffected: there is
no sub-floor regime. The follow-on framing here — "two thirds of requests fall below the shortest
length timed" — turned out to be a units error rather than a real gap: those were *payload*
lengths compared against *rendered* anchors. Measured consistently, the median request renders to
144 tokens and **no** request falls below the floor, because the floor is the empty prompt.

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

| Stage-1 regime | model | A (published) | B (drift) | C (5-anchor) | drift | anchor | **D (current)** |
|---|---|---|---|---|---|---|---|
| nf4 | qwen2.5-1.5b | -4.0% | -5.4% | -12.5% | -1.4 pt | **-7.1 pt** | **-2.2%** |
| nf4 | llama3.2-1b | +4.4% | +3.6% | -2.5% | -0.8 pt | **-6.2 pt** | **+2.3%** |
| bf16 | qwen2.5-1.5b | +16.3% | +15.9% | +11.0% | -0.4 pt | **-4.9 pt** | **+17.4%** |
| bf16 | llama3.2-1b | +21.5% | +21.8% | +16.7% | +0.3 pt | **-5.1 pt** | **+20.0%** |

The anchor effect (-4.9 to -7.1 pt) is roughly six times the session drift (±1 pt).

⚠️ **Column C is no longer the current number; column D is.** Two later changes moved it. Adding
192/256/384 anchors on the re-timed curve recovered 1.6 to 2.9 points, mostly because the corrected
74-token floor (§1) is cheaper than the 86-token one this session measured. Then a unit bug in the
same script was fixed: it had been looking each request's *raw payload* length up on a curve whose
x-axis is the *rendered prompt* length, which understated `k2` more than `k1` and cost a further
2.1 to 7.2 points. Column D is the regenerated result (`results/analysis/cost_lwfull_*.json`,
2026-09-02); see `FINDINGS_cost_vs_length.md` §4.

**This overturns the section heading above.** Correcting the floor moved the result against the
cascade; correcting the units moved it back by more. Net against the published column A the four
arms end up within ±2.1 points (+1.8, −2.1, +1.1, −1.5) — so the large swing this section
reported was mostly a measurement error that has since cancelled out. The surviving claim is
narrower, and it is about precision rather than length: **nf4 Stage-1 is roughly cost-neutral on
this traffic (+2.3% / −2.2%) and bf16 Stage-1 saves 17–20%.**

**Why it moves against the cascade.** The baseline pays k2 on *every* input, so
cheapening k2 at the floor cheapens the baseline faster than it cheapens the cascade,
whose cost is dominated by k1 running on all 25,747 inputs. The naive expectation
(lower k2 helps the cascade) is backwards.

## 4. The sign depends on a Stage-1 quantization choice

nf4 Stage-1 is cost-negative on length-weighted traffic; bf16 Stage-1 is clearly positive.
The cause is that 4-bit dequantization overhead dominates at batch=1 for a 1-2B model:

| model | k1 @512, nf4 | k1 @512, bf16 | re-timed nf4 | re-timed bf16 |
|---|---|---|---|---|
| llama3.2-1b | 109.9 ms | 96.1 ms | 111.4 ms | 98.5 ms |
| qwen2.5-1.5b | 185.8 ms | 138.6 ms | 184.1 ms | **157.6 ms** |

⚠️ **The 138.6 ms figure does not reproduce.** Two independent processes in the 2026-09-02 session
both measured qwen bf16 at 512 tokens as 157.3-157.6 ms (sd 0.07), so 138.6 was an outlier rather
than session drift. It inflated the published bf16 qwen uniform-512 reduction, which falls from
+36.8% to +33.3%. The conclusion of this section still holds with the corrected value: bf16
Stage-1 is 12-14% cheaper than nf4 at 512 tokens and 34-36% cheaper at 128, where the
dequantization overhead is the whole cost.

A 1-2B Stage 1 does not need 4-bit to fit on a 23 GB card, so **bf16 is the correct
deployment regime** and 4-bit was a training-time convenience. The published
length-weighted figures quote the nf4 arm, i.e. the worse of the two.

**The uniform-512 headline is unaffected in shape**, though its values were re-measured: as
recorded here, llama 28.2% / qwen 26.2% (nf4) and llama 30.8% / qwen 34.8% (bf16). The 2026-09-02
re-timing gives **llama 27.9% / qwen 26.4% (nf4), llama 37.5% / qwen 33.3% (bf16)** — the nf4 arms
barely move, and the bf16 arms move because they now use each precision's own measured escalation
rate (0.444 / 0.377) instead of reusing nf4's (0.516 / 0.397).

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
   and neither is Llama's +4.4%" resolves only in part. Both nf4 arms land within ±2.3% of zero
   (§3 column D), so their *sign* is still not firmly established; what is established is that
   nf4 is cost-neutral and bf16 saves 17-20%. Replace the hedge with that, and name the regime.
2. `sec:exp-uncertainty` - drop the positional-join caveat (now verified) and rewrite the
   drift sentence as a quantization-regime difference, not checkpoint provenance.
3. `sec:exp-limitations` - "the shortest length timed is 128 tokens, below which two thirds
   of inputs fall" should be **deleted, not corrected**: the floor is 71-75 rendered tokens (§1),
   is timed directly, and no input falls below it. The limitation does not exist.
4. Consider reporting the bf16 Stage-1 arm as the deployment configuration.

## 8. Not addressed

Batched throughput remains unmeasured; everything here is batch=1.
