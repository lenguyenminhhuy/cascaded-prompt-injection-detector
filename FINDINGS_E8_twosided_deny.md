# E8 — Two-sided Stage-1 (adding the `unsafe → DENY` branch)

**Question.** The architecture diagram draws a two-sided Stage-1 (`unsafe → DENY`,
`safe → ALLOW`, `uncertain → escalate`), but the headline cascade runs *one-sided*
(auto-pass benign only; never blocks). Does adding the DENY branch help?

**Method.** Re-score the frozen Stage-1 + M2 eval logits (no GPU, payload-safe).
Hold overall FPR = 1%. Add a second threshold `p_deny`: escalated rows with
`p_safe ≤ p_deny` are blocked at Stage-1. Stage-1 benign-denials are unfixable FPs
that spend the shared 1% budget, so M2's `theta2` is re-solved on the remainder.
`p_deny` is selected on cal and frozen (deployable), and an eval-optimal DENY point
is also reported as an in-sample upper bound. Script: `scripts/eval_cascade_twosided.py`.
Output: `results/analysis/cascade_{qwen2,llama3}/twosided_summary.json`.

## Result — the DENY branch does not help at the operating point

| Stage-1 | Routing | DR@1%FPR | Eval FPR | Escalation | S1 attacks blocked | S1 benign wrongly blocked |
|---|---|---:|---:|---:|---:|---:|
| Qwen-1.5B | one-sided (headline) | **0.4152** | 0.99% | 39.7% | 0 | 0 |
| Qwen-1.5B | two-sided, strict 1% FPR (in-sample UB) | 0.4152 | 1.00% | 39.7% | 1 | 0 |
| Qwen-1.5B | two-sided, cal-frozen (deployable) | 0.5895 | **5.84%** ✗ | 19.6% | 4088 | 1088 |
| Llama-1B | one-sided (headline) | **0.4012** | 0.98% | 51.6% | 0 | 0 |
| Llama-1B | two-sided, strict 1% FPR (in-sample UB) | 0.4012 | 1.00% | 51.6% | 1 | 0 |
| Llama-1B | two-sided, cal-frozen (deployable) | 0.6577 | **10.84%** ✗ | 25.6% | 4675 | 2020 |

**Two findings, both models:**

1. **At a strictly-held 1% FPR, the DENY branch adds exactly 0 detection** (in-sample upper
   bound: +0 attacks, ΔDR = 0.0000, both models — `twosided_summary.json`
   `primary_verdict.at_strict_1pct_fpr`). To block attacks at Stage-1 you must set a deny threshold,
   but Stage-1 cannot cleanly isolate attacks *above the benign tail* — essentially
   no attacks are separable at zero benign cost. So the deny branch buys nothing the
   1% budget can afford. **This empirically confirms the one-sided design (decision W1).**

2. **A deployable (cal-frozen) deny threshold does not transfer OOD.** It looks great
   in-sample (DR 0.59 / 0.66) but on eval it blows the FPR budget to **5.84% / 10.84%**,
   driven by 1,088 / 2,020 benign inputs wrongly blocked at Stage-1 that cannot be
   re-tuned. This is the **same OOD threshold-transfer failure** as the E6e calibration
   bound — and it is *worse* for a DENY branch, because Stage-1 FPs are irreversible
   (M2 never sees them).

**Cost.** The deny branch *does* cut M2 calls (escalation 39.7%→19.6% Qwen), and the
saving grows with attack prevalence (at 90% attacks it saves ~52 pp of M2 calls). But
that saving only exists at the inflated FPR above; at a strictly-held 1% FPR the deny
branch is ~off, so its cost saving collapses to the one-sided value. **The deny branch
trades false-positive budget for M2 cost — not free.**

## Takeaway for the thesis / diagram

The two-sided architecture is a valid *design*, but at this operating point the
`unsafe → DENY` branch is **empirically not budget-positive** — it adds no detection
at 1% FPR and does not hold the FPR budget OOD. The diagram should show Stage-1 as
one-sided (auto-pass or escalate), with the DENY branch marked as an available-but-
disabled design option, and cite this ablation as the reason. Root cause: Stage-1 is
too miscalibrated OOD to be trusted with an irreversible block.
