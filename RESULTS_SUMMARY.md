# Results Summary — Coupled Face-Body Motion Modeling

**Date:** 2026-08-21
**Author:** Youssef Husseiny Fathy Maaod
**Status:** Core finding established and cross-validated. Not yet written up as a paper.

## Research question

Does an explicit, learned face-body temporal coupling model capture real correspondence between
facial expression and upper-body motion — beyond what either signal alone, or motion smoothness
alone, would predict?

## Method (brief)

- **Data:** BEAT2 (English subset) — 1,620 synchronized SMPL-X body + FLAME facial expression
  sequences across 25 speakers. AMASS/CMU (1,983 sequences) used for auxiliary body-motion
  pretraining context.
- **Models compared (ablation):**
  - **A1** — predicts upper-body motion from its own recent history only (no face input)
  - **A2** — predicts upper-body motion from facial expression only (no temporal history)
  - **A3 (proposed)** — predicts upper-body motion from both facial expression *and* temporal
    history, via a GRU-based probabilistic coupling model
- **Evaluation:** 5-fold cross-validation with subject-level held-out splits (no subject ever
  appears in both train and validation for a given split), so every result below is on speakers
  none of the three models trained on.

## Core finding

**A3 reliably distinguishes a subject's real facial motion from a random mismatched one, at
71.0% ± 3.8% accuracy across 5 independent held-out splits (chance = 50%).**

| Model | Discrimination accuracy (mean ± std, 5 splits) | Interpretation |
|---|---|---|
| A1 (history, no face) | 50.0% (tied by construction) | Cannot discriminate — has no access to facial signal |
| A2 (face, no history) | 20.7% ± 28.4% (highly unstable) | Learned essentially no usable face conditioning; output collapses to near-constant regardless of input |
| **A3 (proposed)** | **71.0% ± 3.8%** | Consistently, robustly above chance across every split |

*(See attached chart: `discrimination_accuracy_summary.png`)*

**What "discrimination accuracy" means:** for each held-out motion window, we test the model
against the subject's real facial track and a randomly mismatched one from a different window/
subject. If the model has learned real face-body structure, it should predict body motion more
accurately with the correct face than the wrong one. A3 does this correctly ~7 times out of 10;
a model with no real understanding would do so exactly half the time.

**Statistical significance:** a single split's result (71%, ~9,000 held-out windows) corresponds
to a z-score of roughly 12 against the chance baseline — this is not noise. The consistency across
5 independently-split, non-overlapping subject groups (std of only 3.8 percentage points) rules out
this being a fluke of one lucky train/val partition.

## What this does *not* yet show

To keep this honest and avoid overclaiming to the team:

- **Raw next-frame prediction accuracy (RMSE) does not clearly favor A3 over simpler baselines.**
  At 30fps, frame-to-frame motion is smooth enough that even a trivial "predict no change"
  baseline is competitive on RMSE. This is why we moved to the discrimination test above — it's a
  sharper test of *whether real correspondence was learned*, separate from general motion
  smoothness.
- **Free-running (autoregressive) generation is not yet reliable.** When the model predicts many
  frames ahead using only its own prior predictions (no grounding), error compounds and results
  degrade. The model currently works well when given real context at every step (its intended use
  as a regularizer/discriminator), not yet as a standalone motion generator.
- **Scope:** single dataset (BEAT2 English), 25 total subjects, gaze inputs are currently
  placeholder values (no gaze dataset integrated yet).
- **A known model limitation:** the coupling model's predicted uncertainty consistently collapses
  to its minimum value during training. This doesn't affect the discrimination result above (which
  only uses the model's mean prediction), but means the "uncertainty-aware" aspect of the design
  isn't functioning as intended yet.

## Attached / available files
- `discrimination_accuracy_summary.png` — chart of the core result above
- `cross_validation_results.json` — full raw per-split numbers (in `outputs/cv/` in the project)
- `PROGRESS_CHECKLIST.md` — detailed internal task/implementation tracker, for reference

## Suggested next steps
1. Team review of this finding before committing to a full paper draft
2. Decide whether to invest in more cross-validation splits / additional datasets for a stronger
   claim, or proceed to writing with the current evidence
3. If proceeding: Method + Results sections should center on the discrimination result as the
   primary claim, with RMSE/rollout results reported as secondary characterizations and limitations
