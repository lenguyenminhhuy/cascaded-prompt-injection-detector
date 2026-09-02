# E10 — Seen-vs-unseen document diagnostic: generalization gap, not data shortage

**Question.** M2 catches only ~5% of eval document injections. Is that (a) a
cross-corpus *generalization* gap (data-diversity lever) or (b) the document/indirect
channel being fundamentally hard even in-distribution (paradigm lever)?

**Method.** M2 was already scored on the held-out validation split, which is drawn
from the *same corpora as training* (BIPIA, synthetic, struq, …). So val document
attacks are SEEN-corpus, held-out instances; eval document attacks (openpromptinjection)
are UNSEEN-corpus. Compare M2's standalone detection (threshold per set on its own
benign). Script: `scripts/diag_seen_vs_unseen.py`. No GPU.

**Leakage control.** Verified by text SHA-256 that **0 of 5,724 val texts appear in
M2's 50,349-row training set** — val is a true holdout, so the seen-side result is
generalization, not memorization.

## Result — decisive

| Set | Channel | n attacks | DR@1% FPR | DR@5% FPR | AUROC vs benign |
|---|---|---:|---:|---:|---:|
| **val (SEEN corpora)** | document | 1,951 | **1.000** | 1.000 | **1.000** |
| | direct | 899 | 1.000 | 1.000 | 1.000 |
| | tool | 258 | 1.000 | 1.000 | 1.000 |
| **eval (UNSEEN corpora)** | document | 2,819 | **0.054** | 0.395 | 0.875 |
| | direct | 2,830 | 0.922 | 0.999 | 0.998 |
| | tool | 1,464 | 0.076 | 0.566 | 0.925 |

M2 detects **seen-corpus** document injections **perfectly** (DR 100%, AUROC 1.00) yet
**collapses** on an **unseen-corpus** document injection set (DR 5.4%). It was trained
on 15,089 document-embedded attacks (the largest channel) — so this is **not** a
coverage gap, and **not** the document channel being inherently undetectable. It is a
**cross-corpus generalization gap**: the model learned the training corpora's
surface cues perfectly and they do not transfer to a new corpus.

**Nuance.** Unseen-document AUROC is 0.875 (well above chance) but DR@1%FPR is only
5.4% while DR@5%FPR is 39.5%. So the signal is *not gone* on unseen documents — it is
too weak / miscalibrated to clear the strict 1%-FPR bar. Some detection is recoverable
by calibration or a channel-specific threshold, but the fundamental fix is diversity /
paradigm.

## Implications for "more data or what?"

1. **More document data of the same corpora will not help** — 15k already gave perfect
   in-corpus detection and 5% out-of-corpus.
2. **Data lever that could help: cross-corpus DIVERSITY** (many injection corpora/styles
   + hard negatives) to break surface-cue shortcuts. Caveat: 2 document corpora already
   failed to cover a 3rd, so breadth must be large; novel-corpus generalization remains
   the open problem.
3. **Paradigm lever (likely higher payoff): span-level detection or task-drift /
   behavioral detection**, which do not depend on attack surface form and transfer
   across corpora better. Known-answer/spotlighting variants worth an ablation.
4. **Cheap partial win:** because unseen-document AUROC is 0.875, per-channel threshold
   calibration or an OOD-representative calibration set recovers some document detection
   without new data.

## Caveats

- Channel and source are 1:1 on this benchmark (document=openpromptinjection, etc.), so
  the channel effect is confounded with a corpus effect; strictly this shows a
  *corpus* generalization gap that happens to fall along channel lines.
- Val benign channels (application-structured, conversational) differ from eval benign;
  per-set 1%-FPR thresholds make absolute DRs set-relative. AUROC-vs-benign is the
  robust threshold-free comparison and tells the same story (1.00 seen vs 0.875 unseen).
- The near-perfect seen-side (all val benign scored exactly 0.0) itself indicates strong
  in-corpus surface regularities the model exploits — consistent with the shortcut story.
