# Runbook — measured cost-vs-length curve

## Why

The thesis says 512-token inputs save more than shorter inputs. The saved fraction is

    reduction(L) = 1 - k1(L)/k2(L) - e

k1 and k2 were measured only at 64 / 128 / 512 / 1024 / 2048 tokens. Everything between
128 and 512 was interpolated, and that band holds 25.0% of the evaluation traffic. This
run measures it.

It also settles a second thing. Derived from the anchors we already have, the saving does
**not** grow monotonically with length — it peaks at 512–1024 and then falls back, because
above roughly 512 tokens Stage 1 stops being fixed-cost and starts scaling with length just
like Stage 2, so k1/k2 stops shrinking. The measured mid-grid will show whether the turn is
where the interpolation puts it.

## What to run (on the A10G box, from `experiments/cascade-pid/`)

Both regimes. `<S2_ADAPTER>` is the Stage-2 LoRA adapter — it is not in this repo, only its
scored logits are. Without it you measure the bare base model, which is faster, which biases
k2 down and understates every saving.

```bash
# Stage-1 NF4 / Stage-2 NF4
PYTHONPATH=. python scripts/benchmark_latency_curve.py \
    --regime nf4 \
    --targets 128,192,256,384,512 \
    --stage1 llama3.2-1b:results_kaggle/stage1/llama3.2-1b/adapter \
    --stage1 qwen2.5-1.5b:results_kaggle/stage1/qwen2.5-1.5b/adapter \
    --stage2 mistral-7b-v0.1:<S2_ADAPTER> \
    --out results/analysis/latency_curve_midgrid_nf4.json

# Stage-1 bf16 / Stage-2 NF4
PYTHONPATH=. python scripts/benchmark_latency_curve.py \
    --regime bf16 \
    --targets 128,192,256,384,512 \
    --stage1 llama3.2-1b:results_kaggle/stage1/llama3.2-1b/adapter \
    --stage1 qwen2.5-1.5b:results_kaggle/stage1/qwen2.5-1.5b/adapter \
    --stage2 mistral-7b-v0.1:<S2_ADAPTER> \
    --out results/analysis/latency_curve_midgrid_bf16stage1.json
```

128 and 512 are re-measured on purpose. They already exist in the old curves, so they are the
drift control: they tell you whether this session is comparable to the one the thesis quotes.

Expect roughly 10–15 minutes per regime. Most of it is model loading; the timing itself is
3 models × 5 lengths × 48 forwards.

## Then, locally

```bash
python scripts/analyze_uniform_length_cost.py \
    --extra-curve nf4=results/analysis/latency_curve_midgrid_nf4.json \
    --extra-curve bf16=results/analysis/latency_curve_midgrid_bf16stage1.json \
    --out results/analysis/cost_uniform_length_sweep.json
```

It prints a drift check first. Read that before reading anything else.

## How to read the result

**Drift.** The re-measured 128 and 512 points should land within about 1% of the old file,
and slightly high rather than slightly low — the new harness hits the target token count
almost exactly, whereas the old session undershot (its 512 target rendered to 505 tokens).
A gap much beyond a couple of percent means the two sessions are not comparable, and the
thesis should quote the new file end to end rather than a mixture.

**The claim.** "512 saves more than shorter inputs" holds if reduction at 512 is above
reduction at 128, 192, 256 and 384 in all four arms. On the interpolated numbers it holds by
a wide margin, and two arms are actually cost-*negative* at 128 tokens — the cascade there
costs more than running Stage 2 alone.

**The other claim.** "the saving grows with request length" is the one at risk. If reduction
at 512 exceeds reduction at 1024 and 2048, the saving peaks and then declines, and the thesis
wording at lines 1784–1785, 1926 and 1972 needs to say "peaks around 512–1024 tokens" instead
of "grows with length". Note this is about the *percentage* saved. Absolute dollars saved
keep rising with length ($51/M at 512 to $143/M at 2048 for llama NF4), so the architecture
still pays off most on long inputs — just not by a growing fraction.

## Acceptance

- both `latency_curve_midgrid_*.json` exist, `gpu` reads `NVIDIA A10G`, `reps` is 40
- `drift_check` in the sweep output shows overlapping targets within tolerance
- `cost_uniform_length_sweep.json` has points at 192, 256 and 384 in all four arms
- thesis wording reconciled against the measured curve
