"""
Multi-step open-loop rollout evaluation on the FULL held-out validation set.

Two corrections vs. the first version of this script:
1. Uses the SAME subject-level train/val split as train_prior.py (same seed, same val_fraction)
   so evaluation only ever touches the 2 subjects the models never trained on. The first version
   evaluated on one random batch drawn from ALL 25 subjects, including the 23 used for training --
   not a valid held-out test.
2. Aggregates RMSE correctly across the whole val set: accumulates the raw sum of squared errors
   and element count across every batch, then takes ONE square root at the end. Averaging
   per-batch RMSE values (what the first version effectively did with one batch) is a subtly
   different and less correct quantity once you have more than one batch.

Usage:
    python scripts/eval_ablations.py --converted_dir ./data/beat2/converted/beat_english_v2.0.0 \
        --checkpoint_dir ./outputs/checkpoints --context_len 10 --horizons 1 5 10 20 40
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
from train_prior import split_files_by_subject  # reuse the EXACT same split logic used in training


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
def rollout_sqerr_and_count(model, ablation: str, z_face, gaze, z_upper, context_len: int, horizons: list[int]):
    """
    Returns, for each horizon h: (sum_of_squared_errors, element_count) accumulated over this
    BATCH only. Caller sums these across all batches, then takes sqrt(total_sq / total_count)
    ONCE at the very end for a statistically correct global RMSE.
    """
    B, T, _ = z_upper.shape
    max_horizon = max(horizons)
    assert context_len + max_horizon <= T

    sqerr_per_step = []   # scalar sum of squared error over (B, UPPER_DIM), per predicted step
    count_per_step = []    # element count (B * UPPER_DIM) per step -- same every step, kept explicit

    z_upper_prev_full = torch.cat([z_upper[:, :1], z_upper[:, :-1]], dim=1)  # matches training convention

    if ablation == "A2":
        # A2 has no history input; reconstruct against the TRUE previous frame each step (it was
        # never designed to consume its own or another model's rolled-forward state).
        for t in range(context_len, context_len + max_horizon):
            pred_delta = model(z_face[:, t:t+1], gaze[:, t:t+1]).squeeze(1)
            abs_pred = reconstruct_absolute(z_upper[:, t - 1], pred_delta)
            diff = abs_pred - z_upper[:, t]
            sqerr_per_step.append((diff ** 2).sum().item())
            count_per_step.append(diff.numel())

    elif ablation == "A1":
        # Burn in hidden state on the shifted ground-truth history, then feed back own predictions.
        # Model input at every step remains the ABSOLUTE previous frame (ground truth during
        # burn-in, this model's own reconstructed absolute prediction during rollout) -- only the
        # model's OUTPUT is a delta that must be reconstructed before use.
        _, _, h = model(z_upper_prev_full[:, :context_len])
        prev_abs = z_upper[:, context_len - 1:context_len]  # (B, 1, UPPER_DIM), absolute
        for t in range(context_len, context_len + max_horizon):
            mean, log_var, h = model(prev_abs, h0=h)
            delta_pred = mean[:, 0]
            abs_pred = reconstruct_absolute(prev_abs[:, 0], delta_pred)
            diff = abs_pred - z_upper[:, t]
            sqerr_per_step.append((diff ** 2).sum().item())
            count_per_step.append(diff.numel())
            prev_abs = abs_pred.unsqueeze(1)

    elif ablation == "A3":
        _, _, h = model(z_face[:, :context_len], gaze[:, :context_len], z_upper_prev_full[:, :context_len])
        prev_abs = z_upper[:, context_len - 1:context_len]
        for t in range(context_len, context_len + max_horizon):
            mean, log_var, h = model(z_face[:, t:t+1], gaze[:, t:t+1], prev_abs, h0=h)
            delta_pred = mean[:, 0]
            abs_pred = reconstruct_absolute(prev_abs[:, 0], delta_pred)
            diff = abs_pred - z_upper[:, t]
            sqerr_per_step.append((diff ** 2).sum().item())
            count_per_step.append(diff.numel())
            prev_abs = abs_pred.unsqueeze(1)

    else:
        raise ValueError(ablation)

    results = {}
    for h_step in horizons:
        results[h_step] = (sum(sqerr_per_step[:h_step]), sum(count_per_step[:h_step]))
    return results


@torch.no_grad()
def persistence_sqerr_and_count(z_upper, context_len: int, horizons: list[int]):
    frozen = z_upper[:, context_len - 1]
    sqerr_per_step, count_per_step = [], []
    for t in range(context_len, context_len + max(horizons)):
        diff = frozen - z_upper[:, t]
        sqerr_per_step.append((diff ** 2).sum().item())
        count_per_step.append(diff.numel())

    results = {}
    for h_step in horizons:
        results[h_step] = (sum(sqerr_per_step[:h_step]), sum(count_per_step[:h_step]))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--converted_dir", required=True)
    parser.add_argument("--checkpoint_dir", default="./outputs/checkpoints")
    parser.add_argument("--context_len", type=int, default=10)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20, 40])
    parser.add_argument("--window_length", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_fraction", type=float, default=0.1, help="MUST match train_prior.py's value")
    parser.add_argument("--seed", type=int, default=42, help="MUST match train_prior.py's value")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use the IDENTICAL split as training, so we only ever evaluate on subjects the models never saw.
    _, val_files = split_files_by_subject(args.converted_dir, args.val_fraction, args.seed)
    ds = CanonicalSequenceDataset(files=val_files, window_length=args.window_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"Evaluating on the FULL held-out val set: {len(ds)} windows across {len(val_files)} files\n")

    models = {}
    for ablation in ["A1", "A2", "A3"]:
        m = load_checkpoint(ablation, args.checkpoint_dir, device)
        if m is not None:
            models[ablation] = m

    # accumulators[name][horizon] = [sum_sq, count]
    accumulators = {name: {h: [0.0, 0] for h in args.horizons} for name in ["Persistence"] + list(models.keys())}

    n_batches = 0
    for batch in loader:
        theta, psi, gaze = batch["theta"].to(device), batch["psi"].to(device), batch["gaze"].to(device)
        z_face = extract_face_descriptor(theta, psi)
        z_upper = extract_upper_descriptor(theta)

        for h_step, (sq, cnt) in persistence_sqerr_and_count(z_upper, args.context_len, args.horizons).items():
            accumulators["Persistence"][h_step][0] += sq
            accumulators["Persistence"][h_step][1] += cnt

        for name, model in models.items():
            for h_step, (sq, cnt) in rollout_sqerr_and_count(
                model, name, z_face, gaze, z_upper, args.context_len, args.horizons
            ).items():
                accumulators[name][h_step][0] += sq
                accumulators[name][h_step][1] += cnt

        n_batches += 1

    print(f"Aggregated over {n_batches} batches, {len(ds)} total windows.\n")

    header = f"{'Model':<24}" + "".join(f"h={h:<8}" for h in args.horizons)
    print(header)
    print("-" * len(header))
    for name, per_horizon in accumulators.items():
        row = f"{name:<24}"
        for h_step in args.horizons:
            sq, cnt = per_horizon[h_step]
            rmse = (sq / cnt) ** 0.5 if cnt > 0 else float("nan")
            row += f"{rmse:<10.4f}"
        print(row)


if __name__ == "__main__":
    main()
