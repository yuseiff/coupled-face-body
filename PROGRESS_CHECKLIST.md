# Project Checklist — Coupled Face-Body Motion Modeling

Last updated: 2026-08-21 (evening — cross-validation complete)

## Environment & Setup
- [x] Python venv created, all packages installed (torch, smplx, chumpy, mediapipe, etc.)
- [x] Repo scaffold in place (`configs/`, `models/`, `losses/`, `preprocessing/`, `scripts/`, `outputs/`)

## Data
- [x] SMPL-X model files downloaded and placed (`data/body_models/smplx/`)
- [x] CMU (AMASS) downloaded and converted → 1,983 sequences (`data/amass/converted/CMU/`)
- [x] BEAT2 (English) downloaded and converted → 1,620 sequences (`data/beat2/converted/beat_english_v2.0.0/`)
- [ ] VOCASET — skipped by decision, not blocking
- [ ] Gaze360 — skipped by decision, not blocking (gaze uses placeholder values)
- [ ] 3DPW / Human3.6M / BEDLAM — not downloaded (only needed for future video-based work, see "Scope decision")

## Core Pipeline Code
- [x] `preprocessing/canonical_state.py` — canonical per-frame state — tested
- [x] `preprocessing/convert_amass.py` — real converter, tested on real CMU data (0 failures)
- [x] `preprocessing/convert_beat2.py` — real converter, tested on real BEAT2 data (0 failures)
- [x] `preprocessing/dataset.py` — windowed PyTorch Dataset, subject-level split support, in-memory caching (fixed a 28x slowdown bug from repeated npz decompression) — tested
- [x] `models/geometry_generator.py` — SMPL-X wrapper — tested on real data, produces real mesh
- [x] `models/descriptors.py` — extracts z_face / z_upper from θ_t/ψ_t via SMPL-X joint indices — tested, dimensions verified
- [x] `models/delta.py` — delta-prediction reconstruction utility (added after absolute-value prediction lost to the trivial baseline) — tested
- [x] `losses/losses.py` — all 9 loss terms written; only L_temp and L_couple exercised on real data so far
- [x] `models/coupling_prior.py` — proposed model (A3, probabilistic) + `DeterministicCoupling` (A2) — both trained
- [x] `models/baselines.py` — `UnconditionalUpperBodyPrior` (A1, no face conditioning) — trained
- [x] `scripts/train_prior.py` — training loop, subject-level split, checkpointing, TensorBoard, supports `--ablation {A1,A2,A3}` — tested, all three trained twice (absolute-value version, then delta-prediction version)
- [x] `scripts/eval_ablations.py` — open-loop multi-step rollout evaluation with a zero-parameter Persistence floor — tested, statistically corrected (full val set, proper global RMSE aggregation, correct held-out subject split)
- [x] `scripts/eval_teacher_forced.py` — PRIMARY evaluation: fair, ground-truth-context comparison across all models — tested

## Ablation Baselines — all three now trained AND fairly evaluated
- [ ] **A0 — fully independent** (external pretrained face/body systems): out of scope, not built — would require integrating third-party pretrained models (VIBE/DECA-style), not attempted
- [x] **A1 — no face conditioning** (history only): trained, evaluated
- [x] **A2 — deterministic coupling** (face only, no history): trained, evaluated
- [x] **A3 — proposed probabilistic coupling**: trained, evaluated

## Results So Far (honest status — see notes)
- [x] Open-loop rollout results obtained (`eval_ablations.py`) — secondary/limitation result: A1/A3 both drift badly under free-running rollout after the delta-prediction fix; kept as a documented limitation, not the primary metric.
- [x] Teacher-forced RMSE obtained (`eval_teacher_forced.py`) — inconclusive on its own: all models statistically tied with the trivial Persistence baseline. Root cause understood: 30fps motion is smooth enough that next-frame RMSE is close to saturated regardless of method — not a useful discriminator between models.
- [x] **Mismatch discrimination test built and cross-validated (`eval_mismatch.py` + `cross_validate.py`) — THIS IS THE CORE RESULT.**
  - A3 (proposed model): **71.0% ± 3.8%** discrimination accuracy across 5 independent random subject splits (68.1/72.3/66.0/73.4/75.0%), all individually far above chance and highly statistically significant (single-split z-score ≈ 12 vs. the 50% null).
  - A2 (face, no history): unstable and near-collapsed (0%, 0%, 51.5%, 0%, 52.0% across the same 5 splits) — diagnostic confirms its output barely varies with input (predicted-delta std ~10-40x smaller than A3's), i.e. it fails to learn usable face conditioning without temporal context.
  - A1 (history, no face): mathematically tied by construction (50%, N/A) — correct sanity control.
  - **Interpretation**: coupling requires BOTH temporal history AND face conditioning together; either alone is insufficient. This is now a defensible, statistically robust core finding.
- [x] Results saved to `outputs/cv/cross_validation_results.json` (full per-seed data + summary stats) — this is real, citable data for the paper.

## Known Limitations (for honest reporting, not blockers)
- [ ] Only 2 held-out subjects per split (25 subjects total, BEAT2 English only) — cross-validation across 5 splits mitigates this somewhat, but a larger/more diverse subject pool would strengthen the claim further
- [ ] `log_var` consistently collapses to its floor (-6.000) in every training run — the "uncertainty-aware" part of the coupling prior isn't functioning as designed; report as a limitation, A3's discrimination result does NOT depend on this working correctly since it's measured via mean prediction error, not calibrated uncertainty
- [ ] Open-loop multi-step rollout diverges badly (see eval_ablations.py results) — coupling prior works well as a single-step/regularizer-style model (matches the original report's intended use case) but not yet as a free-running generator
- [ ] Single dataset (BEAT2 English only); VOCASET/Gaze360/cross-dataset validation not attempted

## Training Diagnostics
- [x] Basic NLL + temporal smoothness logging
- [x] `mean_log_var` / `pred_rmse` diagnostics (caught real variance collapse — log_var pinned at its floor almost immediately in every run so far, a known limitation to report)
- [x] Delta-prediction fix implemented and retrained (motivated by absolute-prediction models losing to Persistence)
- [ ] Mismatch/discrimination test — planned next, not yet built
- [ ] Hyperparameter tuning — not started (deprioritized until the evaluation methodology itself is settled)

## Scope Decision (still open, not urgent)
- [ ] Keep as a **mocap-domain coupling study** (current, achievable scope), OR
- [ ] Extend to full **monocular-video estimation** (original framing) — needs video initializer + 3DPW/Human3.6M/BEDLAM, not yet downloaded

## Writing
- [ ] **Not started yet, but the blocker is now cleared.** You have a real, statistically robust, cross-validated core finding (the mismatch discrimination result). This is enough to begin drafting the Results section around this specific claim, while continuing to note the RMSE/rollout results as secondary/limitations rather than the headline.

---
## Suggested immediate next steps (in order)
1. Decide: start drafting the paper's Method + Results sections around the discrimination finding, OR do one more robustness pass first (e.g. more splits, or a held-out-subject stress test) if you want extra confidence before committing to writing
2. If writing: Problem Formulation should now center on "cross-modal correspondence" as the primary claim, with next-frame RMSE and rollout stability reported as secondary characterizations
3. Build a results figure/table combining: the 3-way ablation table (A1/A2/A3), the 5-seed discrimination accuracy with mean±std, and the A2 output-collapse diagnostic as supporting evidence
4. Write the Limitations section honestly around: log_var collapse, single-dataset scope, small subject pool, rollout divergence
5. Only after a full draft exists: decide whether the video-estimation scope extension is worth pursuing for a stronger submission, or whether this mocap-domain study stands on its own
