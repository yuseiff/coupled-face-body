"""
Face-body MISMATCH DISCRIMINATION test -- the sharpest test of whether the proposed coupling
mechanism has learned real face-body correspondence, rather than just "motion is smooth" (which
is all the RMSE-based tests so far have really been measuring -- see eval_teacher_forced.py's
result, where every model tied with a zero-parameter baseline).

Idea (standard in audio-visual sync-detection literature, applied here to face/body):
For each window in the val set, build a MATCHED pair (a subject's real face motion with their own
real body motion) and a MISMATCHED pair (that same body motion, but paired with a DIFFERENT
subject/window's face motion, via a batch-roll). If a model has learned real face-body structure,
it should predict body motion noticeably WORSE (higher error / lower likelihood) when given the
wrong face track than the right one. If it hasn't, matched and mismatched performance will be
statistically indistinguishable.

A1 is included only as a built-in sanity control: since it never sees z_face/gaze at all, its
error is mathematically IDENTICAL under matched and mismatched conditions by construction -- if
this script ever showed a difference for A1, that would indicate a bug, not a finding.

Two things are reported per model (A2, A3):
    1. Global RMSE under matched vs. mismatched conditions (aggregated correctly across the
       whole val set, one sqrt at the end -- same convention as the other eval scripts).
    2. Per-window discrimination accuracy: the fraction of windows where the matched pairing gave
       LOWER error than the mismatched pairing. Chance level is 50%.

Usage:
    python scripts/eval_mismatch.py --converted_dir ./data/beat2/converted/beat_english_v2.0.0 \
        --checkpoint_dir ./outputs/checkpoints
"""
import argparse
import json
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
def per_window_mse(model, ablation: str, z_face, gaze, z_upper, z_upper_prev) -> torch.Tensor:
    """Returns (B,) -- mean squared reconstruction error per window (averaged over T and UPPER_DIM).
    z_upper/z_upper_prev are always the window's OWN true body sequence; only z_face/gaze may be
    swapped by the caller to test the matched vs. mismatched condition."""
    if ablation == "A1":
        mean, _, _ = model(z_upper_prev)          # never uses z_face/gaze -- see module docstring
    elif ablation == "A2":
        mean = model(z_face, gaze)
    elif ablation == "A3":
        mean, _, _ = model(z_face, gaze, z_upper_prev)
    else:
        raise ValueError(ablation)

    abs_pred = reconstruct_absolute(z_upper_prev, mean)
    sq_err = (abs_pred - z_upper) ** 2               # (B, T, UPPER_DIM)
    return sq_err.mean(dim=(1, 2))                     # (B,) -- per-window mean squared error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--converted_dir", required=True)
    parser.add_argument("--checkpoint_dir", default="./outputs/checkpoints")
    parser.add_argument("--window_length", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64,
                         help="Must be >=2; mismatched pairing is built by rolling the batch by 1.")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="MUST match train_prior.py")
    parser.add_argument("--seed", type=int, default=42, help="MUST match train_prior.py")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    assert args.batch_size >= 2, "batch_size must be >= 2 to construct a mismatched pairing"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, val_files = split_files_by_subject(args.converted_dir, args.val_fraction, args.seed)
    ds = CanonicalSequenceDataset(files=val_files, window_length=args.window_length)
    # drop_last=True so every batch has a full, well-defined roll-by-1 pairing
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"Evaluating on held-out val set: {len(ds)} windows across {len(val_files)} files\n")

    models = {}
    for ablation in ["A1", "A2", "A3"]:
        m = load_checkpoint(ablation, args.checkpoint_dir, device)
        if m is not None:
            models[ablation] = m

    # accumulators[name]['matched'/'mismatched'] = [sum_sq_of_per_window_mse... ] -- we track sums
    # of per-window MSE (not raw element-wise sums here, since discrimination accuracy needs
    # per-window values anyway) plus window counts, and separately a running count of "matched
    # pairing won" for the discrimination accuracy.
    stats = {name: {"matched_sum": 0.0, "mismatched_sum": 0.0, "n_windows": 0, "matched_wins": 0}
              for name in models}

    n_batches = 0
    for batch in loader:
        theta, psi, gaze = batch["theta"].to(device), batch["psi"].to(device), batch["gaze"].to(device)
        z_face = extract_face_descriptor(theta, psi)
        z_upper = extract_upper_descriptor(theta)
        z_upper_prev = torch.cat([z_upper[:, :1], z_upper[:, :-1]], dim=1)

        # mismatched pairing: roll face/gaze by 1 along the batch dimension, so window i's body
        # is paired with window (i-1)'s face/gaze instead of its own. With shuffle=True on the
        # DataLoader, which windows land next to each other changes every batch/epoch.
        z_face_mismatched = torch.roll(z_face, shifts=1, dims=0)
        gaze_mismatched = torch.roll(gaze, shifts=1, dims=0)

        for name, model in models.items():
            matched_mse = per_window_mse(model, name, z_face, gaze, z_upper, z_upper_prev)              # (B,)
            mismatched_mse = per_window_mse(model, name, z_face_mismatched, gaze_mismatched, z_upper, z_upper_prev)

            stats[name]["matched_sum"] += matched_mse.sum().item()
            stats[name]["mismatched_sum"] += mismatched_mse.sum().item()
            stats[name]["n_windows"] += matched_mse.shape[0]
            stats[name]["matched_wins"] += (matched_mse < mismatched_mse).sum().item()

            # Diagnostic (first batch only): is this model's output actually varying with its
            # input, or has it collapsed to a near-constant prediction regardless of z_face?
            # A near-zero std here would explain an extreme, non-noise-like discrimination score
            # without needing to invoke any real (mis)coupling behavior.
            if n_batches == 0:
                with torch.no_grad():
                    if name == "A2":
                        pred = model(z_face, gaze)
                    elif name == "A3":
                        pred, _, _ = model(z_face, gaze, z_upper_prev)
                    else:
                        pred = None
                    if pred is not None:
                        print(f"  [diagnostic] {name} predicted-delta std across batch: "
                              f"{pred.std().item():.6f} (near-zero = output barely depends on input)")

        n_batches += 1

    print(f"Aggregated over {n_batches} batches.\n")
    print(f"{'Model':<8}{'Matched RMSE':<16}{'Mismatched RMSE':<18}{'Discrimination acc.':<22}{'Note'}")
    print("-" * 90)
    summary = {}
    for name, s in stats.items():
        matched_rmse = (s["matched_sum"] / s["n_windows"]) ** 0.5
        mismatched_rmse = (s["mismatched_sum"] / s["n_windows"]) ** 0.5
        disc_acc = s["matched_wins"] / s["n_windows"]

        if name == "A1":
            # By construction, A1's matched/mismatched errors are bitwise identical (it never
            # consumes z_face/gaze), so "matched < mismatched" is never strictly true -- this
            # would otherwise misleadingly print as 0.0%. Report it plainly as a tie instead.
            acc_str, note = "N/A (tied)", "control: never sees face by design"
            summary[name] = {"matched_rmse": matched_rmse, "mismatched_rmse": mismatched_rmse,
                              "discrimination_acc": None, "n_windows": s["n_windows"]}
        else:
            acc_str = f"{disc_acc*100:.1f}%"
            note = "chance level (~50%)" if abs(disc_acc - 0.5) < 0.02 else ""
            summary[name] = {"matched_rmse": matched_rmse, "mismatched_rmse": mismatched_rmse,
                              "discrimination_acc": disc_acc, "n_windows": s["n_windows"]}

        print(f"{name:<8}{matched_rmse:<16.4f}{mismatched_rmse:<18.4f}{acc_str:<22}{note}")

    # Machine-readable line for scripts/cross_validate.py to parse from captured stdout, without
    # disturbing the human-readable table above.
    print("RESULT_JSON:" + json.dumps(summary))


if __name__ == "__main__":
    main()
