# Coupled Latent-State Modeling of Facial Expression, Gaze, Head Pose, and Upper-Body Motion

## Research question
Does an explicit learned face-body temporal coupling prior capture real correspondence between
facial expression and upper-body motion, beyond what smoothness or either signal alone predicts?

## Core finding (see `RESULTS_SUMMARY.md` for full details)
The proposed coupled model (A3) distinguishes a subject's real facial motion from a randomly
mismatched one at **71.0% ± 3.8% accuracy** across 5 independent held-out subject splits (chance
= 50%). Ablations without either temporal history (A2) or facial input (A1) do not show this
ability. See `RESULTS_SUMMARY.md` and `outputs/discrimination_accuracy_summary.png`.

## Repo layout
```
configs/          YAML configs (dataset paths, loss weights, ablation IDs)
data/              raw + converted datasets (NOT tracked in git -- see Setup below)
preprocessing/     dataset -> canonical sequence converters, PyTorch Dataset
models/            geometry generator, descriptor extraction, coupling prior + baselines
losses/            individual loss terms
scripts/           train_prior.py, eval_teacher_forced.py, eval_mismatch.py, eval_ablations.py,
                   cross_validate.py
outputs/           checkpoints/logs (NOT tracked), cross_validation_results.json (tracked)
```

## Setup
1. `python -m venv venv` then activate it, `pip install -r requirements.txt`
2. Download datasets per `DATASET_ACCESS.md` (SMPL-X, AMASS/CMU, BEAT2 -- none are redistributed
   here due to license restrictions)
3. Convert data: `python preprocessing/convert_amass.py ...` / `convert_beat2.py ...`
4. Train: `python scripts/train_prior.py --converted_dir <path> --ablation {A1,A2,A3}`
5. Evaluate: `python scripts/eval_mismatch.py --converted_dir <path> --checkpoint_dir outputs/checkpoints`
6. Reproduce the cross-validated core result: `python scripts/cross_validate.py --converted_dir <path> --n_splits 5 --epochs 10`

## Key documents
- `RESULTS_SUMMARY.md` — current results, written for a team/external audience
- `PROGRESS_CHECKLIST.md` — detailed internal implementation/task tracker
- `DATASET_ACCESS.md` — where to get each dataset and what's required

## Status
Core discrimination finding established and cross-validated (5 splits). Not yet submitted/written
as a full paper. See `PROGRESS_CHECKLIST.md` for outstanding items and known limitations.
