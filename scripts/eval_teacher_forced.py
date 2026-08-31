"""
PRIMARY evaluation: teacher-forced, full-sequence RMSE on the held-out val set.

Unlike scripts/eval_ablations.py (open-loop rollout, useful as a secondary "how far does this
drift unassisted" stress test / limitation), this script evaluates every model the way it was
actually trained AND the way the implementation report says it's meant to be deployed (Section
7.3: "if intended for estimation, use its negative log-likelihood as a regularizer") -- i.e. with
real ground-truth context (z_upper_prev, z_face_t, gaze_t) available at every single step, no
autoregressive feedback. This is the fair, primary comparison for the paper:
    - It removes A2's structural advantage in the rollout test (A2 never used its own predictions
      as history to begin with, so open-loop rollout wasn't a fair test of it either way).
    - It matches the actual training objective exactly, so these numbers are directly interpretable.

Aggregates correctly across the WHOLE held-out val set (same subject-level split as training),
accumulating raw squared error and count, then taking one sqrt at the end -- not averaging
per-batch RMSE values.

Usage:
    python scripts/eval_teacher_forced.py \
        --converted_dir ./data/beat2/converted/beat_english_v2.0.0 \
        --checkpoint_dir ./outputs/checkpoints
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from preprocessing.dataset import CanonicalSequenceDataset
from models.coupling_prior import ProbabilisticCouplingPrior, DeterministicCoupling
from models.baselines import UnconditionalUpperBodyPrior
from models.descriptors import extract_face_descriptor, extract_upper_descriptor, FACE_DIM, UPPER_DIM
from models.delta import reconstruct_absolute
from train_prior import split_files_by_subject


def load_checkpoint(ablation: str, checkpoint_dir: str, device: str):
    path = os.path.join(checkpoint_dir, f"{ablation}_best.pt")
    if not os.path.exists(path):
        print(f"WARNING: no checkpoint found for {ablation} at {path}, skipping.")
        return None
    if ablation == "A1":
        model = UnconditionalUpperBodyPrior(upper_dim=UPPER_DIM)
    elif ablation == "A2":
        model = DeterministicCoupling(face_dim=FACE_DIM, gaze_dim=3, upper_dim=UPPER_DIM)
    elif ablation == "A3":
        model = ProbabilisticCouplingPrior(face_dim=FACE_DIM, gaze_dim=3, upper_dim=UPPER_DIM)
    else:
        raise ValueError(ablation)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded {ablation} checkpoint from epoch {ckpt['epoch']} (val_nll={ckpt['val_nll']:.4f})")
    return model


@torch.no_grad()
def teacher_forced_sqerr_and_count(model, ablation: str, z_face, gaze, z_upper, z_upper_prev):
    """Every input is ground truth at every step -- no autoregression. Returns (sum_sq, count)
    over the WHOLE batch (all T frames), for later aggregation across the full val set."""
    if ablation == "A2":
        pred_delta = model(z_face, gaze)                       # (B, T, UPPER_DIM)
    else:  # A1, A3 both output (mean, log_var, hidden)
        if ablation == "A1":
            mean, _, _ = model(z_upper_prev)
        elif ablation == "A3":
            mean, _, _ = model(z_face, gaze, z_upper_prev)
        pred_delta = mean

    abs_pred = reconstruct_absolute(z_upper_prev, pred_delta)
    diff = abs_pred - z_upper
    return (diff ** 2).sum().item(), diff.numel()


@torch.no_grad()
def persistence_sqerr_and_count(z_upper, z_upper_prev):
    """Zero-parameter floor: predicted delta = 0, i.e. predicted_absolute = z_upper_prev."""
    diff = z_upper_prev - z_upper
    return (diff ** 2).sum().item(), diff.numel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--converted_dir", required=True)
    parser.add_argument("--checkpoint_dir", default="./outputs/checkpoints")
    parser.add_argument("--window_length", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_fraction", type=float, default=0.1, help="MUST match train_prior.py")
    parser.add_argument("--seed", type=int, default=42, help="MUST match train_prior.py")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, val_files = split_files_by_subject(args.converted_dir, args.val_fraction, args.seed)
    ds = CanonicalSequenceDataset(files=val_files, window_length=args.window_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"Evaluating on the FULL held-out val set: {len(ds)} windows across {len(val_files)} files\n")

    models = {}
    for ablation in ["A1", "A2", "A3"]:
        m = load_checkpoint(ablation, args.checkpoint_dir, device)
        if m is not None:
            models[ablation] = m

    accumulators = {name: [0.0, 0] for name in ["Persistence"] + list(models.keys())}

    n_batches = 0
    for batch in loader:
        theta, psi, gaze = batch["theta"].to(device), batch["psi"].to(device), batch["gaze"].to(device)
        z_face = extract_face_descriptor(theta, psi)
        z_upper = extract_upper_descriptor(theta)
        z_upper_prev = torch.cat([z_upper[:, :1], z_upper[:, :-1]], dim=1)

        sq, cnt = persistence_sqerr_and_count(z_upper, z_upper_prev)
        accumulators["Persistence"][0] += sq
        accumulators["Persistence"][1] += cnt

        for name, model in models.items():
            sq, cnt = teacher_forced_sqerr_and_count(model, name, z_face, gaze, z_upper, z_upper_prev)
            accumulators[name][0] += sq
            accumulators[name][1] += cnt

        n_batches += 1

    print(f"Aggregated over {n_batches} batches, {len(ds)} total windows, "
          f"{ds.window_length} frames/window.\n")

    print(f"{'Model':<24}{'RMSE (all frames, teacher-forced)':<20}")
    print("-" * 55)
    for name, (sq, cnt) in accumulators.items():
        rmse = (sq / cnt) ** 0.5 if cnt > 0 else float("nan")
        print(f"{name:<24}{rmse:<20.4f}")


if __name__ == "__main__":
    main()
