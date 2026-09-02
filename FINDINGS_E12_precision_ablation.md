# E12 — Stage-1 precision ablation: bf16 vs NF4

**Experiment:** score Stage-1 in both bf16 and 4-bit NF4 on the same GPU in the same session, so
the thesis's bf16 rows rest on measurement instead of on NF4's escalation rate
**Hardware:** NVIDIA A10G, batch=1. Stage-2 always NF4.
**Status:** complete — all four Stage-1 passes done, plus a matched NF4 control
**Artifacts:** `results/analysis/E12_precision_ablation.json`,
`results/stage1_prec/{nf4,bf16}/{llama3.2-1b,qwen2.5-1.5b}/eval_logits.jsonl`
**Payload-safe:** labels, logits and aggregates only.

## 1. Why this was needed

Table 6's bf16 Stage-1 block reported a cost reduction, but bf16 detection had never been scored
and the rows reused the escalation rate `e` measured under NF4. Since `e` is what the cost model
is most sensitive to (`FINDINGS_E11_stage1_score_sensitivity.md`), the bf16 half of the headline
rested on an unmeasured quantity. This experiment measures it.

Stage 2 never re-runs: M2's per-input eval logits were already saved, so all four passes are
Stage-1 scoring plus an offline cascade replay.

## 2. The confound that had to be cleared first

The pre-existing NF4 logits came from Kaggle (P100) while the latency constants came from an
A10G, and the box's PEFT version was older than the one that trained the adapters. A bf16-vs-NF4
delta measured across that gap would confound precision with environment, so a **matched NF4
control** was scored on the same A10G in the same session.

**It passed bit-exactly.** NF4-on-A10G is byte-identical to the paper source for both arms
(qwen md5 `9198a1e7…`, llama md5 `578efd18…`), and the cascade replay reproduces
`e` = 0.3969 / 0.5161, DR = 0.4152 / 0.4012 and leaked = 460 / 639 exactly. The version and
hardware confound is eliminated: every delta below is precision alone.

## 3. Stage-1 detection alone (DR@1%FPR on the 25,747-row eval)

| system | DR@1%FPR | note |
|---|---|---|
| qwen2.5-1.5b **bf16** | **0.3640** | measured A10G |
| llama3.2-1b **bf16** | 0.2709 | measured A10G |
| qwen2.5-1.5b NF4 | 0.1709 | paper / Kaggle P100, reproduced bit-exactly |
| llama3.2-1b NF4 | 0.1890 | paper / Kaggle P100, reproduced bit-exactly |
| ProtectAI DeBERTa (encoder baseline) | 0.2931 | |
| PromptGuard-2 86M (encoder baseline) | 0.2210 | |
| M2 Mistral-7B (Stage 2 alone) | 0.4015 | |

**Quantization was costing Stage-1 roughly half its detection.** bf16 more than doubles qwen's
solo DR (0.1709 → 0.3640) and lifts llama's by 8.2 points. This overturns a claim in the thesis:
§4.3's statement that no fine-tuned Stage-1 beats the encoder baselines **holds only under NF4** —
bf16 qwen at 0.3640 exceeds ProtectAI's 0.2931 by 7.1 points and lands within 3.8 points of the
7B Stage-2.

## 4. Cascade at the cal-frozen operating point (θ_safe = 0.005)

| precision | Stage-1 | e | DR | FPR | leaked | gap CI vs M2 | verdict |
|---|---|---|---|---|---|---|---|
| NF4 | qwen2.5-1.5b | 0.3969 | 0.4152 | — | 460 | — | EXCEEDS |
| NF4 | llama3.2-1b | 0.5161 | 0.4012 | — | 639 | — | MATCHES |
| bf16 | qwen2.5-1.5b | **0.3772** | **0.4218** | 0.0097 | 513 | [+0.0119, +0.0337] | EXCEEDS |
| bf16 | llama3.2-1b | **0.4442** | **0.4146** | 0.0099 | 962 | [+0.0016, +0.0208] | EXCEEDS |

Two results here. **The measured bf16 `e` is lower than NF4's** (0.3772 vs 0.3969; 0.4442 vs
0.5161), so reusing NF4's `e` had *understated* bf16's cost saving — the opposite of what the
E11 sensitivity analysis warned might happen. And **bf16 llama moves from MATCHES to EXCEEDS**:
its bootstrap gap CI against M2-alone clears zero.

**The safety cost is real and lands on llama.** Its auto-passed (leaked) attacks rise
639 → 962, i.e. 9.0% → **13.5%** of the 7,113 attacks, purely from the precision change. DR rises
at the same time, which is exactly the pattern E11 flagged: a better-separating Stage-1 auto-passes
more traffic, so the 1%-FPR budget concentrates on a smaller escalated pool and DR improves while
irrecoverable false negatives get worse. Total FN still passes the pre-registered criterion
(58.5% vs M2's 59.9%), but DR alone would have hidden this.

## 5. Cost: use the re-measured numbers, not this artifact's

`E12_precision_ablation.json` carries cost figures computed with the 2026-08-20 latency constants
(k1 179.5 / 138.3 / 108.9 / 95.8 ms, k2 543.5 ms). **Those are superseded** by the 2026-09-01
eight-length re-timing (`FINDINGS_cost_vs_length.md`), which also found the old qwen-bf16 512-token
anchor of 138.6 ms to be an unreproducible outlier — it is 157.6 ms. The detection results above
are unaffected; only the cost columns move.

| precision | Stage-1 | uniform-512, this artifact | **uniform-512, current** | length-weighted, this artifact | **length-weighted, current** |
|---|---|---|---|---|---|
| NF4 | qwen2.5-1.5b | 27.3% | **26.4%** | −4.0% | **−2.2%** |
| NF4 | llama3.2-1b | 28.4% | **27.9%** | +4.4% | **+2.3%** |
| bf16 | qwen2.5-1.5b | 36.8% | **33.3%** | +16.3% | **+17.4%** |
| bf16 | llama3.2-1b | 38.0% | **37.5%** | +21.5% | **+20.0%** |

(The length-weighted column also absorbs a unit fix in `analyze_length_weighted_cost.py`, which had
been pricing requests by raw payload length on a rendered-prompt curve; see
`FINDINGS_cost_vs_length.md` §4.)

The experiment's headline survives the re-measurement intact, and in fact sharpens: **bf16 Stage-1
is what makes the cascade pay for itself on realistic traffic.** On length-weighted accounting both
NF4 arms sit within ±2.3% of zero while both bf16 arms save 17–20%, and qwen's break-even margin
goes from none at all under NF4 (break-even 0.449 against a realised `e` of 0.397 — barely clear)
to comfortable under bf16 (0.616 against 0.377).

## 6. What bf16 costs

Peak VRAM roughly doubles the adapter-time footprint but stays small in absolute terms:
llama 3.71 → 5.16 GB, qwen 4.35 → 6.29 GB. A 1–1.5B Stage-1 does not need 4-bit to fit on a
23 GB card, so NF4 was a training-time convenience carried into inference by habit rather than a
deployment requirement.

## 7. Recommended configuration

**Qwen2.5-1.5B bf16.** It has the best detection of any Stage-1 (solo DR 0.3640), the best cascade
DR (0.4218, +2.0 points over M2 alone with the CI clear of zero), the lowest leak of the four arms
(513 = 7.2% of attacks), and a length-weighted cost reduction of +17.4%.

Llama bf16 is slightly cheaper (+20.0% length-weighted) but leaks 13.5% of attacks against qwen's
7.2%. That is the trade to state explicitly: 2.6 points of extra cost saving in exchange for
roughly 450 more attacks passed without any chance of being caught. On those terms qwen is the
clear pick — the cost gap is small and the safety gap is not.

## 8. Open caveats

1. **The short-length region is where most of the gain lives, and it is now measured.** bf16 gains
   most at short lengths (qwen 173.6 → 111.1 ms at 128 tokens), and the eval's median request is
   144 rendered tokens. The earlier "short lengths not measured, box retired" caveat is closed: the
   2026-09-02 session timed the template floor directly (75 / 74 / 71 tokens), and since that floor
   *is* the empty prompt, no request falls below the measured curve. Only 0.06% run past its
   2049-token top.
2. **Single-stream only.** Everything is batch=1. Batching amortises the dequantization overhead
   that makes NF4 slow at short lengths, so it should narrow the bf16-vs-NF4 gap; unmeasured.
3. **The 1%-FPR operating point is still eval-calibrated.** This ablation changes Stage-1's
   precision, not the OOD threshold-transfer problem (E6e), which is untouched.
