# Findings — Cascade Cost/Escalation Frontier (E2 Stage-1 → route → modeled Stage-2)

_Last updated: 2026-08-05_

## 0. TL;DR (the headline)

The thesis claims a two-stage cascade (cheap Stage-1 small model + expensive strong
Stage-2) is **cost-effective** — Stage-1 clears easy traffic so it skips Stage-2.
Cost is saved **only** on traffic Stage-1 confidently *and correctly* resolves.

**Finding: on shifted (held-out) data, the current Qwen Stage-1 cannot save meaningful
cost at a safe operating point — even if Stage-2 were perfect.** To reach zero
attack-leakage **and** a ≤1% false-positive rate, Stage-1 must escalate **83–100%** of
traffic, at which point the cascade saves **at most +6.9%** and on two of three splits
is **10% *more* expensive** than calling the strong detector on everything.

This is a **defensible, nuanced negative result**, not a bug. It says: the cascade's
cost-saving premise holds only when Stage-1 is well-calibrated on the traffic it sees;
under distribution shift this Stage-1 collapses, so the cascade contribution has to come
from a *better-calibrated* Stage-1 (motivates E4 calibration) — not from the current one.

> ⚠️ All numbers below were computed on the **repo-bundled** Stage-1 adapter, which
> differs ~6% from the confirmed training-run adapter. They establish the *shape and
> direction* of the result reliably; **re-score the confirmed adapter (Track B) before
> quoting exact figures** in the thesis. See §6.

---

## 1. What was measured, and on what data

- **Stage-1 model:** Qwen2.5-1.5B-Instruct + a LoRA adapter (the injection-detection
  skill; 35 MB diff on the frozen 3 GB base). Emits `p_safe` per input.
- **Splits (honest held-out):**
  | split | n | attacks | benign | what it tests |
  |---|---|---|---|---|
  | `test_in_dist` | 523 | 78 | 445 | same distribution as train |
  | `test_cross_channel` | 366 | 183 | 183 | new delivery channel |
  | `test_cross_domain` | 253 | 149 | 104 | new domain |
- **val / cal** are same-pool-as-train and **saturated** (AUROC ≈ 1.0) — not honest
  generalization estimates; used only for calibration/anchoring.
- **Payload hygiene:** the whole analysis reads only labels + logits (`p_safe`), never
  raw attack text (per repo CLAUDE.md).

---

## 2. Stage-1 standalone metrics — the overfitting story

Rank-based metrics (AUROC robust to miscalibration; ECE = calibration error; DR@1% =
detection rate at 1% false-positive rate):

| split | AUROC | acc@0.5 | DR@1%FPR | ECE | FPR@0.5 |
|---|---|---|---|---|---|
| val/cal (saturated) | 1.000 | 1.000 | 1.000 | 0.0003 | ~0 |
| test_in_dist | 0.818 | 0.491 | 0.167 | **0.484** | 0.573 |
| test_cross_channel | 0.896 | 0.691 | 0.492 | **0.270** | 0.590 |
| test_cross_domain | 0.931 | 0.589 | 0.228 | **0.376** | 1.000 |

Reading:
- **Ranking is decent** (AUROC 0.82–0.93) but **calibration collapses** under shift
  (ECE jumps from ~0.0005 to 0.27–0.48). The model can *order* attacks vs benign, but its
  probabilities are badly wrong.
- **accuracy@0.5 is misleading** here: e.g. cross_domain acc 0.589 while AUROC 0.931 — the
  model pushes *everything* below 0.5 (FPR@0.5 = 1.000, it flags 100% of benign). This is
  why E2 selection was re-pointed onto **AUROC (primary) + ECE (tie-break)**, not accuracy.
- **val AUROC = 1.0 is a corpus artifact** ("training-corpus-style ⇒ safe" shortcut), not
  real skill; it degrades sharply off-distribution.

---

## 3. The cost model (why AUROC alone doesn't decide the thesis)

Cascade expected cost per request: **`C(e) = k1 + e·k2`**
- `k1` = Stage-1 cost, `k2` = Stage-2 cost, `e` = escalation rate (fraction sent to Stage-2).
- Cascade cheaper than Stage-2-only **iff `e < 1 − k1/k2`**.
- Cost reduction: `x = 1 − (k1 + e·k2)/k2`.

Savings come **only** from traffic Stage-1 *auto-resolves* (pass/block without escalating).
Two-sided routing: `p_safe ≥ θ_safe → pass`, `p_safe ≤ θ_unsafe → block`, else **escalate**.
So the quantity that decides the thesis is the **escalation-rate vs. end-to-end-quality
frontier on shifted data** — not Stage-1 AUROC.

---

## 4. The frontier results (modeled Stage-2, with 95% bootstrap CIs)

Stage-2 is **modeled as perfect** (upper bound: escalated attacks all caught, escalated
benign all cleared) — this is the *most generous possible* assumption. If the cascade
fails here, it fails with any real Stage-2.

### 4a. Config grid (θ_safe ∈ [0.50,0.95], θ_unsafe ∈ [0.05,0.50]) — best FPR at zero attack-leak

| split | best end-to-end FPR (95% CI) | escalation e (95% CI) | note |
|---|---|---|---|
| test_in_dist | — | — | **no zero-leak point exists**: ≥8 attacks auto-pass even at the widest θ_safe=0.95 |
| test_cross_channel | 0.437 [0.360, 0.510] | 0.156 [0.117, 0.194] | no point reaches FPR ≤ 0.01 |
| test_cross_domain | 0.625 [0.532, 0.713] | 0.154 [0.115, 0.202] | no point reaches FPR ≤ 0.01 |

Within the calibratable grid, the best you can do at zero attack-leak is a **44–62%
false-positive rate** — unusable. (The CIs are tight and nowhere near 0.01, so this is not
small-n noise.)

### 4b. Widened grid (θ_unsafe down to 0.0) — escalation needed for zero-leak **AND** FPR ≤ 1%

| split | escalation e (95% CI) | cost reduction @ k1/k2=0.1 | meets e ≤ 0.25 target? |
|---|---|---|---|
| test_in_dist | **1.000** [1.000, 1.000] | **−10.0%** (cascade costs *more*) | ✗ |
| test_cross_channel | **0.831** [0.795, 0.869] | **+6.9%** | ✗ |
| test_cross_domain | **1.000** [1.000, 1.000] | **−10.0%** (cascade costs *more*) | ✗ |

**Mechanism:** under shift, 44–62% of *benign* traffic lands at `p_safe ≤ 0.05` — Stage-1
confidently *mis*-flags it. To avoid wrongly blocking those benign you must escalate them,
which drives `e` to 0.83–1.0. At `e → 1` you pay `k1` on top of a full Stage-2 call, so the
cascade is strictly worse than Stage-2-only. The proposal's `e ≤ 25%` target is missed on
every split.

---

## 5. Interpretation for the thesis

- The cascade is **not** cost-effective with the current Stage-1 under distribution shift.
  This is a **clean, honest result** — and arguably a more interesting thesis contribution
  than a naive "it saves X%" claim that wouldn't survive shift.
- The bottleneck is **calibration, not ranking.** AUROC 0.82–0.93 says the signal exists;
  ECE 0.27–0.48 and FPR@0.5 up to 1.0 say the *thresholds/probabilities* are broken. A
  calibrated Stage-1 (E4) is the natural next contribution: recover cost savings by fixing
  where the benign mass sits, not by training a bigger model.
- Framing to use: *"A cheap Stage-1 only yields cascade cost savings when it is calibrated
  on the deployment distribution; we quantify the escalation/quality frontier and show that
  corpus-artifact overfitting eliminates the savings under shift, motivating explicit
  calibration."*

---

## 6. Critical limitations (must be stated in any writeup)

1. **Stage-2 is MODELED, not measured.** The intended strong detector (Mistral-7B "M2") is
   untrained (empty stub); DataSentinel outputs are on a different dataset and unusable
   (~62–65% FPR). Frontier uses perfect (upper bound) + parameterized Stage-2. Real
   end-to-end numbers await the Stage-2 M2 fine-tune and its validation go/no-go gate.
2. **Cost is unit-invariant; absolute $ is illustrative only.** `k1` (Stage-1 latency) is
   unmeasured and there is no same-GPU `(k1, k2)` pair yet. Reported as
   cost-reduction % vs. the `k1/k2` ratio; only `k2 ≈ 1459 ms` (DataSentinel) is measured.
3. **Thresholds are PRE-calibration (E4).** This is the *exploratory* frontier, not the final
   calibrated operating point.
4. **Small n** (253/366/523). Mitigated with 1000× paired-bootstrap CIs (§4), but still small.
5. **Adapter provenance.** Numbers are from the **repo-bundled** adapter, ~6% off the
   confirmed training-run adapter (`results_kaggle/stage1/qwen2.5-1.5b/adapter/`). Re-score
   the confirmed adapter (Track B) before quoting exact figures. Shape/direction are robust.

---

## 7. Code & artifacts (all local; nothing committed)

| file | status |
|---|---|
| `scripts/sweep_thresholds.py` | **new** — frontier sweep, modeled Stage-2, unit-invariant cost, `--bootstrap N` CIs, `--theta-*-range` grid overrides. Payload-safe. |
| `scripts/rank_stage1.py` | **edited** — E2 selection re-pointed accuracy → AUROC (primary) + ECE (tie-break). |
| `scripts/score_split.py` | **edited** — added `rendered_input` fallback + empty-text guard (test splits store text under `rendered_input`; without this it silently emits constant-`p_safe` garbage). |
| `results/analysis/cost_frontier/` | config-grid frontier + CIs (`frontier_summary.json`, per-split `.json`/`.csv`). |
| `results/analysis/cost_frontier_wide/` | full-tail frontier + CIs. |

Verification passed: anchor pin (θ_safe=θ_unsafe=0.5 → e=0, e2e_fpr = raw Stage-1 FPR@0.5
exactly: 0.573/0.590/1.000); monotonicity; 0 NaN; row counts = split sizes; vectorized
route() cross-checked against scalar `route()`.

**Scope:** this is the `sweep_thresholds.py` half of the threshold-sweep tooling work, and it
feeds the E7 cost-effectiveness analysis. **Not final** — Stage-2 and cost inputs are still
modeled.

---

## 8. Next steps

- **Track A (local, done):** frontier + bootstrap CIs. ✅ Remaining: regenerate the visual
  walkthrough artifact with these CI-backed numbers.
- **Track B (1 GPU step, then local):** re-score the **confirmed** adapter via
  `scripts/score_split.py` on cal + 3 test splits → re-run §4 → confirm adapter-robustness.
  Removes limitation #5.
- **Track C (optional, 2 GPU trainings):** train + score Llama-3.2-1B and Granite-2B so the
  AUROC ranker compares a real candidate field.
- **Unblock real end-to-end (larger):** train Stage-2 M2 (fine-tune, then the validation
  go/no-go gate) and measure same-GPU `(k1, k2)` to replace modeled Stage-2 and illustrative $.
