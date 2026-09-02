# E13 — Are Stage-1 and Stage-2 failures nested or complementary?

**Claim under test:** the attacks Stage 1 cannot catch, Stage 2 cannot catch either.

**Verdict: true for document-embedded and tool-output attacks, false for direct
injection.** The claim holds in the two channels that motivate indirect-injection
detection in the first place, and fails in the one channel whose payloads overlap
training. Reported for all four precision/model arms; the pattern is identical in
every one, so it is not a precision or model artifact.

Artifacts: `scripts/analyze_failure_overlap.py`,
`results/analysis/failure_overlap_{qwen,llama}_{nf4,bf16}.json`,
`scripts/make_figure_failure_overlap.py`,
`results/figures/thesis/failure_overlap.{pdf,png}`.

Operating points: each stage at its own eval 1%-FPR threshold (oracle, as
everywhere in the paper). Stage 2 = Mistral-7B M2, DR 0.4015, threshold 0.2095.
Attacks n=7,113.

## 1. Globally the failures overlap, but are NOT nested

| arm | S1 DR | P(S2 miss) | P(S2 miss \| S1 miss) | lift | phi | S2 rescues of S1 misses | union ceiling |
|---|---|---|---|---|---|---|---|
| qwen-nf4  | 0.1701 | 0.5985 | 0.6756 | 1.13x | 0.347 | 32.4% | 0.4393 |
| qwen-bf16 | 0.3640 | 0.5985 | 0.7655 | 1.28x | 0.450 | 23.5% | 0.5131 |
| llama-nf4 | 0.1874 | 0.5985 | 0.6720 | 1.12x | 0.312 | 32.8% | 0.4540 |
| llama-bf16| 0.2709 | 0.5985 | 0.6745 | 1.13x | 0.254 | 32.5% | 0.5082 |

The lift is only 1.12-1.28x and the union ceiling (0.44-0.51) exceeds S2 alone
(0.4015), so the stages ARE partially complementary overall. **Do not claim the
failure sets are nested globally — the data does not support it.**

## 2. Per channel the picture inverts (Qwen bf16; all arms in the JSONs)

| channel | n | S1 DR | S2 DR | both miss | **S2 rescue of S1 misses** |
|---|---|---|---|---|---|
| direct (struq)   | 2830 | 0.6360 | 0.9205 | 4.0%  | **88.9%** |
| document (OPI)   | 2819 | 0.0255 | 0.0514 | 92.8% | **4.8%**  |
| tool (agentdojo) | 1464 | 0.4898 | 0.0724 | 50.1% | **1.9%**  |

Rescue rate across all four arms: direct 88.9-91.3%, document 4.3-5.1%,
tool 1.9-2.5%. Two things to state in the paper:

- **Document and tool: shared blindness.** Escalating buys almost nothing —
  Stage 2 recovers under 5% of what Stage 1 missed.
- **Tool channel inverts the cascade premise.** Stage 1 *outperforms* Stage 2
  (0.490 vs 0.072 for Qwen bf16). The "strong" stage is the weaker one there,
  so routing to it is actively counterproductive on that channel.

## 3. The cascade's own leak set is near-undetectable by Stage 2

At the frozen theta_safe=0.005, attacks Stage 1 auto-passes:

| arm | leaked | **S2 DR on leaked** | S2 DR on escalated | ratio |
|---|---|---|---|---|
| qwen-nf4  | 460 | 0.0043 | 0.4290 | 100x |
| qwen-bf16 | 513 | 0.0039 | 0.4324 | 111x |
| llama-nf4 | 639 | 0.0031 | 0.4408 | 142x |
| llama-bf16| 962 | 0.0249 | 0.4604 | 18x  |

Stage-2 DR falls monotonically as Stage-1 p_safe rises (Spearman -0.56 qwen-bf16,
-0.44 llama-bf16, both p~0). For Qwen, Stage 2 catches **0 of the 291 attacks**
with Stage-1 p_safe > 0.02 (0/245 for nf4; 0/154 above 0.1; 0/76 above 0.5).

**Why this matters for the paper's argument:** the router is not discarding
catchable attacks. The leak costs almost no detection because what gets leaked is
what Stage 2 would also have missed. This is the strongest available defence of
confidence routing — and simultaneously evidence that the cascade inherits rather
than resolves the shared ceiling, which is what the conclusion already claims.

## 4. Caveats

- Oracle thresholds throughout (eval-set benign distribution). Same limitation as
  every DR@1%FPR number in the paper.
- Channel and source are collinear here: direct=struq_synthetic,
  document=openpromptinjection, tool=agentdojo, 1:1. Channel effects cannot be
  separated from corpus effects, and the paper's own payload-overlap analysis says
  direct's high numbers rest on payloads reused from training. Read "direct" as
  "the familiar-payload slice", not as a claim about delivery channel.
- Section 2's global non-nestedness is driven mostly by the direct slice; the
  per-channel numbers are the substantive result.

## 5. Payload hygiene

`--examples` emits identifiers, scores, char counts and sha256 prefixes only,
never dataset text (repo rule). Example both-miss IDs for qwen-bf16 are in the
JSON; inspect payloads outside an assistant session via
`scripts/inspect_samples.py` or `scripts/demo_pipeline.py --show-payloads`.

## 6. Addendum — does preservation hold per channel?

Checked after the fact, because aggregate DR parity can hide a per-channel
regression. It does not here. Cascade vs M2-on-every-input, both at overall
1% FPR, theta_safe=0.005 frozen on cal (`results/analysis/preservation_by_channel.json`):

| arm | direct | document | tool | ALL |
|---|---|---|---|---|
| qwen-nf4   | +0.0106 | +0.0156 | +0.0280 | +0.0162 |
| qwen-bf16  | +0.0155 | +0.0213 | +0.0383 | +0.0225 |
| llama-nf4  | +0.0011 | +0.0021 | +0.0034 | +0.0020 |
| llama-bf16 | +0.0032 | +0.0153 | +0.0280 | +0.0131 |

**No channel degrades in any arm.** Every delta is positive.

**Why the cascade exceeds its own strong stage** — state this explicitly, because
it otherwise reads as a bug. Auto-passing benign inputs shrinks the benign pool
reaching Stage 2, so the same 1% *global* FPR budget buys a **lower** Stage-2
threshold on the escalated subset (0.187-0.207 vs 0.2095 global). It is FPR-budget
reallocation, not error correction. The cascade does not fix any Stage-2 mistake;
it spends the false-positive allowance on a smaller pool.

## 7. What the preservation claim does and does not cover

Proved: aggregate DR matched-or-exceeded in all four arms (paired bootstrap,
1,000 resamples); per-channel DR non-decreasing in all four arms; FPR held at
0.0097-0.0099; theta_safe frozen on cal, never tuned on eval.

Not proved:
- **Deployable operating point.** Every DR@1%FPR here uses a threshold set on the
  eval set's own benign distribution. Sec 4.6 reports that freezing M2's cal
  threshold gives 57.9% eval FPR. This is a matched *oracle* comparison.
- **Generalisation.** Single corpus, single seed, no replication. The held-out
  document control (1.00 seen vs 0.05 unseen corpus) says the ceiling is fragile.
- **Absolute adequacy.** Preservation is at DR 0.40, not at a useful level.
- **Error structure.** DR parity hides that llama-bf16 leaks 962 attacks (13.5%)
  irrecoverably; M2-on-every-input has no equivalent category.
- **Third candidate.** Granite-2B was never scored in bf16.
