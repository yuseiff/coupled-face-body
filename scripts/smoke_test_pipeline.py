"""
Full pipeline smoke test on REAL converted data.

Chains together every piece built so far:
    dataset.py (real BEAT2 window) -> geometry_generator.py (real mesh) -> descriptors.py
    (z_face, z_upper) -> coupling_prior.py (NLL) -> losses.py (temporal smoothness)

This is the first true "does the whole thing work together" check. It does NOT train anything
(no optimizer step) -- it only confirms that real data flows through every module without shape
errors and produces finite, sane loss values.

Usage:
    python scripts/smoke_test_pipeline.py --body_models_path ./data/body_models \
        --converted_dir ./data/beat2/converted/beat_english_v2.0.0
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # repo root, for package imports below

import torch
from torch.utils.data import DataLoader

from preprocessing.dataset import CanonicalSequenceDataset
from models.geometry_generator import GeometryGenerator
from models.descriptors import extract_face_descriptor, extract_upper_descriptor, FACE_DIM, UPPER_DIM
from models.coupling_prior import ProbabilisticCouplingPrior
from losses.losses import coupling_nll_loss, temporal_smoothness_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--body_models_path", default="./data/body_models")
    parser.add_argument("--converted_dir", required=True)
    parser.add_argument("--window_length", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. real data ---
    ds = CanonicalSequenceDataset(args.converted_dir, window_length=args.window_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    batch = next(iter(loader))
    beta = batch["beta"].to(device)          # (B, 10)
    theta = batch["theta"].to(device)          # (B, T, 165)
    psi = batch["psi"].to(device)               # (B, T, 50)
    gaze = batch["gaze"].to(device)              # (B, T, 3)
    B, T, _ = theta.shape
    print(f"OK [data]: batch loaded, subjects={batch['subject_id']}")
    print(f"           beta {tuple(beta.shape)}, theta {tuple(theta.shape)}, psi {tuple(psi.shape)}")

    # --- 2. geometry generator: run on the FIRST frame of each window as a mesh sanity check ---
    gen = GeometryGenerator(body_models_path=args.body_models_path, device=device).to(device)
    out = gen(beta, theta[:, 0], psi[:, 0])
    print(f"OK [geometry]: vertices {tuple(out.vertices.shape)}, joints {tuple(out.joints.shape)}")
    assert torch.isfinite(out.vertices).all(), "Non-finite values in generated mesh -- check inputs."

    # --- 3. descriptor extraction across the whole window ---
    z_face = extract_face_descriptor(theta, psi)     # (B, T, FACE_DIM)
    z_upper = extract_upper_descriptor(theta)          # (B, T, UPPER_DIM)
    assert z_face.shape == (B, T, FACE_DIM)
    assert z_upper.shape == (B, T, UPPER_DIM)
    print(f"OK [descriptors]: z_face {tuple(z_face.shape)}, z_upper {tuple(z_upper.shape)}")

    # --- 4. coupling prior forward + NLL, teacher-forced with the previous real frame ---
    coupler = ProbabilisticCouplingPrior(face_dim=FACE_DIM, gaze_dim=3, upper_dim=UPPER_DIM).to(device)
    z_upper_prev = torch.cat([z_upper[:, :1], z_upper[:, :-1]], dim=1)  # shift by 1, pad first frame
    mean, log_var, _ = coupler(z_face, gaze, z_upper_prev)
    nll = coupling_nll_loss(mean, log_var, z_upper)
    print(f"OK [coupling]: mean {tuple(mean.shape)}, NLL = {nll.item():.4f}")
    assert torch.isfinite(nll), "Coupling NLL is not finite -- check descriptor scales/normalization."

    # --- 5. temporal smoothness loss on the real theta/psi window ---
    smooth, jerk = temporal_smoothness_loss(theta, psi)
    print(f"OK [temporal]: smoothness = {smooth.item():.4f}, jerk = {jerk.item():.4f}")
    assert torch.isfinite(smooth) and torch.isfinite(jerk)

    print("\nALL CHECKS PASSED: real data flows end-to-end through geometry, descriptors, "
          "coupling prior, and losses with finite outputs.")


if __name__ == "__main__":
    main()
