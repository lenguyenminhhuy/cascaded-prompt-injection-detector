# Cost reduction against input length, measured

**Experiment:** cost reduction vs. input length — measure k1/k2 at 192/256/384 tokens to test the 512-token claim
**Hardware:** NVIDIA A10G, g5 class (`ec2-34-221-165-179`), measured 2026-09-02
**Protocol:** batch=1, median of 40 reps after 8 warm-ups, per-input = `predict()` = 2 label
forwards. Stage~2 always nf4. All three models and both Stage-1 precisions timed in one
session, so no ratio pairs measurements taken at different times.
**Harness:** `scripts/benchmark_latency_curve.py` (new; the earlier curves' harness was
never in the repository, only its JSON outputs)
**Artifacts:** `results/analysis/latency_curve_full_{nf4,bf16}.json`,
`results/analysis/cost_uniform_length_sweep.json`,
`results/analysis/cost_lwfull_*.json` (regenerated 2026-09-02 after the token-count fix in §4;
each file now records its own inputs under `"inputs"`)

## 1. The question

The paper claimed 512-token inputs save more than shorter inputs, and separately that the
saving *grows* with length. Neither had been measured: the anchors were the template floor,
128, 512, 1024 and 2048, so the whole 128-to-512 range — which carries 24.7% of evaluation
traffic — was interpolated across a single 384-token gap.

## 2. The template floor is 71-75 tokens, not 83-87

`FINDINGS_short_input_cost.md` §1 records a floor of 83-87 tokens. That is wrong: it
measured a short padded prompt, not an empty one. Rendering the empty input through
`format_prompt` gives **llama 75, qwen 74, mistral 71** tokens. This session's harness
lands its target-64 row exactly on the floor, so the floor is now timed directly.

## 3. Cost reduction against uniform request length

`reduction(L) = 1 - k1(L)/k2(L) - e`, e held at the value measured for that regime.

| tokens | NF4 Llama | NF4 Qwen | bf16 Llama | bf16 Qwen |
|---|---|---|---|---|
| 71-75 (floor) | -1.1% | -27.0% | +20.9% | +3.0% |
| 128 | +4.0% | -17.2% | +26.4% | +12.4% |
| 192 | +12.9% | -1.3% | +32.0% | +22.3% |
| 256 | +22.4% | +15.4% | **+37.8%** | +32.7% |
| 384 | +25.0% | +21.3% | +36.8% | **+34.2%** |
| 512 | **+27.9%** | +26.4% | +37.5% | +33.3% |
| 1024 | +26.9% | **+28.7%** | +35.5% | +32.5% |
| 2048 | +26.2% | +28.6% | +34.1% | +31.6% |

**"512 saves more than shorter inputs" holds in the two NF4 arms and fails in the two bf16
arms**, where the peak is at 256 and 384 respectively. **"The saving grows with length" is
false in all four.** The curve rises steeply from the floor to roughly 256 tokens and is
flat above it, within about 5 points across the whole 256-2048 range.

The mechanism is visible in the raw latencies: Stage 1 is nearly fixed-cost up to 256
tokens (llama nf4: 94.9 to 101.9 ms) while Stage 2 doubles over the same range (192.0 to
393.1 ms). Past 256 both scale with length together, so `k1/k2` stops falling.

Absolute dollars saved keep rising with length ($50/M at 512 to $150/M at 2048 for llama
nf4) because a roughly constant fraction of a growing bill is a growing amount. Only the
percentage plateaus.

## 4. Length-weighted cost on the benchmark's own traffic

Re-run over the 8-anchor curve (`scripts/analyze_length_weighted_cost.py`):

| regime | model | published | 5-anchor | 8-anchor | **8-anchor, rendered lengths** |
|---|---|---|---|---|---|
| nf4 | llama3.2-1b | +4.4% | -2.5% | -0.8% | **+2.3%** |
| nf4 | qwen2.5-1.5b | -4.0% | -12.5% | -9.4% | **-2.2%** |
| bf16 | llama3.2-1b | +21.5% | +16.7% | +17.9% | **+20.0%** |
| bf16 | qwen2.5-1.5b | +16.3% | +11.0% | +12.7% | **+17.4%** |

Adding the 192/256/384 anchors moved every arm by at most 1.5 points, so the 128-to-512
interpolation was **not** a large error source despite spanning a quarter of the traffic.
Measuring the floor was: it is worth 4.9 to 7.1 points, because the baseline pays `k2` on
every input and so gains more than the cascade from a cheaper short-request `k2`. Reading the
lengths in the right units (next paragraph) was worth another 2.1 to 7.2.

**A unit bug in the 8-anchor column, found afterwards and now fixed.**
`analyze_length_weighted_cost.py` took each request's **raw payload** token count and looked it up
on a curve whose x-axis is the **rendered prompt** length. The two differ by the template floor
(§2: 75 / 74 / 71 tokens), so every request was priced about 74 tokens too far left. Stage 2's
curve is much steeper than Stage 1's, so this understated `k2` more than `k1` and therefore
**understated the reduction**.

The script now renders each request through `format_prompt` and tokenizes it with **each stage's
own** tokenizer at `add_special_tokens=False`, which is precisely what
`benchmark_latency_curve.py` records as `actual_prompt_tokens`. Validation: the empty prompt
renders to 75 / 74 / 71 tokens, matching the curves' shortest anchors exactly. The last column
above is the regenerated result (`cost_lwfull_*.json`, 2026-09-02): every arm improves by 2.1
(bf16 llama) to 7.2 points (nf4 qwen), and nf4 llama crosses from negative to slightly positive.

Two conclusions change wording. The eval's median request is **144 rendered tokens**, not 70 —
still far below 512, so uniform-512 still flatters the cascade. And "nf4 loses money" becomes
**"nf4 is roughly cost-neutral"** (+2.3% / −2.2%): the sign is inside plausible measurement drift,
so the claim to make is that the entire length-weighted saving depends on a bf16 Stage-1.

The fix also **retires the flat-hold caveat**. The curve's floor is the empty-prompt length, so no
request can fall below it, and only 0.06% run past the 2049-token top anchor — essentially every
request is priced on measured ground. The old "68% of requests fall below the shortest measured
length" caveat was an artifact of comparing payload lengths against rendered anchors.

## 5. One old anchor does not reproduce

`latency_curve_bf16stage1.json` records qwen bf16 at 512 tokens as 138.6 ms. Two independent
processes in this session both give **157.3-157.6 ms** (sd 0.07). The old value is an
outlier, not session drift, and it inflated the published bf16 Qwen figure: the uniform-512
reduction falls from +36.8% to +33.3%.

Separately, this instance runs **3-6% slower at 1024-2048 tokens** than the instance the
earlier curves were taken on. The `k1/k2` ratios the cost model consumes moved by under
1.5%, so absolute latencies are instance-specific in a way the ratios are not.

## 6. Applied to `main_cybersecurity_sn.tex`

New Table `tab:latency-length` in §4.4 reporting the full profile; Table `tab:cost` (5.9),
the theta_safe sweep table, §5.9 prose, the uncertainty and criteria paragraphs, the
abstract and the conclusion all rebased onto this session. Backup at
`main_cybersecurity_sn.tex.bak-44`.

## 7. Not addressed

Batched throughput remains unmeasured; everything here is batch=1.
`cost_frontier_double.pdf` is still drawn from the earlier measurement session.
