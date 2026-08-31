"""
Trains the proposed probabilistic coupling prior (models/coupling_prior.py) on real converted
BEAT2 data. This is Stage 5 of the implementation roadmap: "Train the proposed contribution."

IMPORTANT -- subject-level train/val split:
BEAT2 filenames encode subject identity as the leading numeric token, e.g. "9_miranda_1_6_6.npz"
is subject 9. Splitting by FILE (i.e. by individual sequence/window) instead of by SUBJECT would let
the same person's face/body appear in both train and val, silently inflating validation metrics.
This script always splits by subject first, then builds separate datasets from the resulting file lists.

What this script does NOT do (by design, for a first version):
    - Does not run the geometry generator / differentiable rendering -- this pretrains only the
      coupling prior on descriptor sequences, matching Stage 5. Full sequence-level refinement
      (Stage 6) with reprojection/photometric losses is a separate, later script.
    - Does not implement early stopping -- it always trains for --epochs, saving the best checkpoint
      by val NLL. Add early stopping later if you find it overfits early.

Usage:
    python scripts/train_prior.py \
        --converted_dir ./data/beat2/converted/beat_english_v2.0.0 \
        --output_dir ./outputs \
        --epochs 20 --batch_size 32
"""
import argparse
import glob
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from preprocessing.dataset import CanonicalSequenceDataset
from models.coupling_prior import ProbabilisticCouplingPrior, DeterministicCoupling
from models.baselines import UnconditionalUpperBodyPrior
from models.delta import reconstruct_absolute
from models.descriptors import extract_face_descriptor, extract_upper_descriptor, FACE_DIM, UPPER_DIM
from losses.losses import coupling_nll_loss, temporal_smoothness_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subject_id_from_filename(path: str) -> str:
    """BEAT2 filenames look like '9_miranda_1_6_6.npz' -> subject id '9'. Adjust this function if
    you later add a dataset with a different naming convention (e.g. AMASS uses full filename)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("_")[0]


def split_files_by_subject(converted_dir: str, val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    all_files = sorted(glob.glob(os.path.join(converted_dir, "*.npz")))
    if not all_files:
        raise FileNotFoundError(f"No .npz files found in {converted_dir}")

    subjects = sorted(set(subject_id_from_filename(f) for f in all_files))
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n_val_subjects = max(1, round(len(subjects) * val_fraction))
    val_subjects = set(subjects[:n_val_subjects])
    train_subjects = set(subjects[n_val_subjects:])

    train_files = [f for f in all_files if subject_id_from_filename(f) in train_subjects]
    val_files = [f for f in all_files if subject_id_from_filename(f) in val_subjects]

    print(f"Subjects: {len(subjects)} total -> {len(train_subjects)} train / {len(val_subjects)} val")
    print(f"Files:    {len(all_files)} total -> {len(train_files)} train / {len(val_files)} val")
    return train_files, val_files


def build_model(ablation: str, device: str) -> nn.Module:
    if ablation == "A1":
        return UnconditionalUpperBodyPrior(upper_dim=UPPER_DIM).to(device)
    elif ablation == "A2":
        return DeterministicCoupling(face_dim=FACE_DIM, gaze_dim=3, upper_dim=UPPER_DIM).to(device)
    elif ablation == "A3":
        return ProbabilisticCouplingPrior(face_dim=FACE_DIM, gaze_dim=3, upper_dim=UPPER_DIM).to(device)
    else:
        raise ValueError(f"Unknown ablation '{ablation}'. Expected one of: A1, A2, A3.")


def compute_loss(batch, model, ablation: str, device, lambda_temp: float) -> tuple[torch.Tensor, dict]:
    theta = batch["theta"].to(device)   # (B, T, 165)
    psi = batch["psi"].to(device)         # (B, T, 50)
    gaze = batch["gaze"].to(device)        # (B, T, 3)

    z_face = extract_face_descriptor(theta, psi)     # (B, T, FACE_DIM)
    z_upper = extract_upper_descriptor(theta)          # (B, T, UPPER_DIM)
    z_upper_prev = torch.cat([z_upper[:, :1], z_upper[:, :-1]], dim=1)  # teacher-forced shift
    delta_target = z_upper - z_upper_prev  # DELTA prediction target -- see models/delta.py

    if ablation == "A1":
        # Critical control: predicts the DELTA of z_upper_t from ONLY its own history, no face/gaze.
        mean, log_var, _ = model(z_upper_prev)
        nll = coupling_nll_loss(mean, log_var, delta_target)
        with torch.no_grad():
            mean_log_var = log_var.mean().item()
            abs_pred = reconstruct_absolute(z_upper_prev, mean)
            pred_rmse = torch.sqrt(((abs_pred - z_upper) ** 2).mean()).item()
        primary_loss = nll

    elif ablation == "A2":
        # Simple deterministic point-estimate of the DELTA from face/gaze, no history input,
        # no uncertainty. z_upper_prev is used only to reconstruct the absolute value for the
        # reported RMSE -- it is NOT fed into the model, preserving A2's "no history input" design.
        pred_delta = model(z_face, gaze)
        mse = nn.functional.mse_loss(pred_delta, delta_target)
        with torch.no_grad():
            mean_log_var = float("nan")  # not applicable -- deterministic model has no variance head
            abs_pred = reconstruct_absolute(z_upper_prev, pred_delta)
            pred_rmse = torch.sqrt(((abs_pred - z_upper) ** 2).mean()).item()
        nll = mse  # named 'nll' in metrics dict for a uniform schema across ablations; it's really MSE here
        primary_loss = mse

    elif ablation == "A3":
        mean, log_var, _ = model(z_face, gaze, z_upper_prev)
        nll = coupling_nll_loss(mean, log_var, delta_target)
        with torch.no_grad():
            mean_log_var = log_var.mean().item()
            abs_pred = reconstruct_absolute(z_upper_prev, mean)
            pred_rmse = torch.sqrt(((abs_pred - z_upper) ** 2).mean()).item()
        primary_loss = nll

    else:
        raise ValueError(f"Unknown ablation '{ablation}'.")

    smooth, jerk = temporal_smoothness_loss(theta, psi)
    total = primary_loss + lambda_temp * smooth

    metrics = {"nll": nll.item() if torch.is_tensor(nll) else nll, "smooth": smooth.item(),
               "jerk": jerk.item(), "total": total.item(),
               "mean_log_var": mean_log_var, "pred_rmse": pred_rmse}
    return total, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--converted_dir", required=True)
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--ablation", choices=["A1", "A2", "A3"], default="A3",
                         help="A1=no face conditioning (control), A2=deterministic coupling, "
                              "A3=proposed probabilistic coupling")
    parser.add_argument("--window_length", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_temp", type=float, default=0.1)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)  # 0 is safest on Windows
    parser.add_argument("--device", default=None, help="cuda / cpu; auto-detected if not set")
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Ablation: {args.ablation}")

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    log_dir = os.path.join(args.output_dir, "logs", f"train_{args.ablation}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # --- subject-level split ---
    train_files, val_files = split_files_by_subject(args.converted_dir, args.val_fraction, args.seed)

    train_ds = CanonicalSequenceDataset(files=train_files, window_length=args.window_length)
    val_ds = CanonicalSequenceDataset(files=val_files, window_length=args.window_length)
    print(f"Windows: {len(train_ds)} train / {len(val_ds)} val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    # --- model + optimizer ---
    model = build_model(args.ablation, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir=log_dir)

    best_val_nll = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        train_metrics_sum = {"nll": 0.0, "smooth": 0.0, "jerk": 0.0, "total": 0.0,
                              "mean_log_var": 0.0, "pred_rmse": 0.0}
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            loss, metrics = compute_loss(batch, model, args.ablation, device, args.lambda_temp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            for k in train_metrics_sum:
                train_metrics_sum[k] += metrics[k]
            n_batches += 1
            global_step += 1

            if global_step % 50 == 0:
                writer.add_scalar("train/nll_step", metrics["nll"], global_step)

        train_avg = {k: v / max(1, n_batches) for k, v in train_metrics_sum.items()}

        # --- validation ---
        model.eval()
        val_metrics_sum = {"nll": 0.0, "smooth": 0.0, "jerk": 0.0, "total": 0.0,
                            "mean_log_var": 0.0, "pred_rmse": 0.0}
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                _, metrics = compute_loss(batch, model, args.ablation, device, args.lambda_temp)
                for k in val_metrics_sum:
                    val_metrics_sum[k] += metrics[k]
                n_val_batches += 1
        val_avg = {k: v / max(1, n_val_batches) for k, v in val_metrics_sum.items()}

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"train NLL {train_avg['nll']:.4f} rmse {train_avg['pred_rmse']:.4f} logvar {train_avg['mean_log_var']:.3f} | "
              f"val NLL {val_avg['nll']:.4f} rmse {val_avg['pred_rmse']:.4f} logvar {val_avg['mean_log_var']:.3f}")

        writer.add_scalar("train/nll_epoch", train_avg["nll"], epoch)
        writer.add_scalar("val/nll_epoch", val_avg["nll"], epoch)
        writer.add_scalar("val/smooth_epoch", val_avg["smooth"], epoch)
        writer.add_scalar("train/pred_rmse", train_avg["pred_rmse"], epoch)
        writer.add_scalar("val/pred_rmse", val_avg["pred_rmse"], epoch)
        writer.add_scalar("train/mean_log_var", train_avg["mean_log_var"], epoch)
        writer.add_scalar("val/mean_log_var", val_avg["mean_log_var"], epoch)

        # --- checkpointing ---
        last_path = os.path.join(checkpoint_dir, f"{args.ablation}_last.pt")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "val_nll": val_avg["nll"]}, last_path)

        if val_avg["nll"] < best_val_nll:
            best_val_nll = val_avg["nll"]
            best_path = os.path.join(checkpoint_dir, f"{args.ablation}_best.pt")
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "val_nll": val_avg["nll"]}, best_path)
            print(f"  -> new best val NLL {best_val_nll:.4f}, saved to {best_path}")

    writer.close()
    print(f"\nTraining complete. Best val NLL: {best_val_nll:.4f}")
    print(f"Checkpoints in: {checkpoint_dir}")
    print(f"View training curves with: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
