# E9 — Anatomy of the Stage-1 leak: can/should we reduce the 6.5%?

**Setup.** The one-sided cascade auto-passes confident-benign at Stage-1. Some
attacks are auto-passed too ("leaked") and never reach M2: 460/7,113 = 6.5% (Qwen),
639/7,113 = 9.0% (Llama). These are final false negatives. Two questions.

Script: `scripts/analyze_stage1_leak.py` (logits only, no GPU, payload-safe).
Output: `results/analysis/cascade_{qwen2,llama3}/leak_anatomy.json`.

## Q2 — If the leaked attacks HAD reached M2, would M2 catch them? No.

Counterfactual at M2's operating threshold θ2 (M2's score exists on every eval row):

| Stage-1 | Leaked attacks | M2 would catch | M2 would miss | Recoverable |
|---|---:|---:|---:|---:|
| Qwen-1.5B | 460 | **2** | 458 | **0.4%** |
| Llama-1B | 639 | **2** | 637 | **0.3%** |

Leaked attacks have an M2 injection-score **median of 0.000**, and **99.6%** sit
*below* θ2. Stage-1 and M2 fail on the **same inputs**. By channel (Qwen):

| Channel | Leaked | M2 would catch | Recoverable |
|---|---:|---:|---:|
| document | 399 | 0 | 0% |
| tool | 58 | 0 | 0% |
| direct | 3 | 2 | 67% |

**87% of the leak is the document channel, which M2 catches only 6.7% of anyway.**
The leak is almost entirely *unrecoverable*: it is the shared detector ceiling
(indirect/document injection), not a Stage-1 routing mistake. Direct attacks — the
ones M2 catches 93% of — essentially never leak (3 of 2,830).

## Q1 — Should we reduce the leak by escalating more? No (at 1% FPR).

Raising θ_safe shrinks the leak but re-tightens θ2 to hold 1% FPR. Sweep (Qwen):

| θ_safe | Leaked | Escalation (cost) | End-to-end DR @1%FPR |
|---:|---:|---:|---:|
| 0.005 (operating) | 460 | 39.7% | **0.4152** |
| 0.105 | 135 | 47.6% | 0.4038 |
| 0.505 | 73 | 53.7% | 0.4015 |
| 0.955 | 38 | 61.5% | 0.4015 |
| 1.005 (escalate all) | 0 | 100% | 0.4015 |

Reducing the leak **lowers** end-to-end detection (0.4152 → 0.4015 = M2-on-everything)
while **raising** cost (escalation 39.7% → 61.5%). You would pay more and detect less.
The leak is part of *why* the cascade slightly beats M2-everywhere: auto-passing easy
benign frees FP-budget headroom that lets M2 run a looser θ2. Escalating the leaked
(M2-blind) attacks removes that headroom for no detection gain.

## Takeaways

1. **The 6.5% leak looks alarming but is ~99.6% attacks M2 also cannot catch** —
   the same document/indirect-injection ceiling, not a router failure.
2. **Tuning the router (θ_safe) is the wrong lever.** Reducing the leak is
   counterproductive at the operating point (more cost, less detection).
3. **The real lever is the detector, not the cascade.** To recover these attacks you
   must improve M2 / the training data on document/indirect injection (or add a
   channel-specific defense). This is the same conclusion as bound (a) — the document
   channel is the whole ceiling.
4. **Caveat:** channel and source are 1:1 here (openpromptinjection=document,
   agentdojo=tool, struq_synthetic=direct), so the channel effect is confounded with
   a source/distribution effect; the two cannot be fully separated on this benchmark.
