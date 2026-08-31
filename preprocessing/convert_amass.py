"""
Converter: AMASS (SMPL-X format, e.g. CMU) -> canonical Sequence format.

Based on the actual on-disk schema of CMU/*.npz files (verified 2026-08-21):
    gender                  ()          scalar string
    surface_model_type      ()
    mocap_frame_rate         ()          scalar, e.g. 120.0
    mocap_time_length        ()
    trans                    (T, 3)
    poses                     (T, 165)    = [root_orient(3), pose_body(63), pose_jaw(3),
                                              pose_eye(6), pose_hand(90)]  <- matches our theta layout exactly
    betas                      (16,)       AMASS fits 16 shape components; we only keep the first N_BETA=10
    root_orient, pose_body, pose_jaw, pose_eye, pose_hand  -- same data as 'poses', split into pieces
                                                                (we use the pre-concatenated 'poses' field directly)

AMASS has NO facial expression or gaze data -- psi_t and gaze_t are set to zero/placeholder. This dataset
is used ONLY to pretrain the generic body motion prior (Stage 1), never for the coupling experiment itself.

Usage:
    python preprocessing/convert_amass.py --raw_dir data/amass/raw/CMU --out_dir data/amass/converted/CMU
"""
import argparse
import glob
import os

import numpy as np

from canonical_state import Sequence, FrameState, N_BETA, N_ALPHA, N_PSI, N_GAZE, N_DELTA, N_CAM, N_THETA

TARGET_FPS = 30.0


def load_amass_npz(npz_path: str, target_fps: float = TARGET_FPS) -> Sequence:
    data = np.load(npz_path, allow_pickle=True)

    poses = data["poses"]              # (T, 165)
    betas_full = data["betas"]          # (16,)
    src_fps = float(data["mocap_frame_rate"])
    T = poses.shape[0]

    if poses.shape[1] != N_THETA:
        raise ValueError(
            f"{npz_path}: expected theta dim {N_THETA}, got {poses.shape[1]}. "
            "This AMASS sub-dataset may use a different pose layout than CMU -- inspect before converting."
        )

    # --- resample to TARGET_FPS by simple stride subsampling ---
    # NOTE: this is a first-pass approximation (nearest-frame subsampling, not interpolation).
    # Replace with scipy.interpolate per-channel later if motion looks choppy after resampling.
    stride = max(1, round(src_fps / target_fps))
    frame_indices = np.arange(0, T, stride)

    beta = betas_full[:N_BETA].astype(np.float32)
    alpha = np.zeros(N_ALPHA, dtype=np.float32)  # AMASS has no face identity data

    frames = []
    for i in frame_indices:
        theta = poses[i].astype(np.float32)
        frames.append(
            FrameState(
                theta=theta,
                psi=np.zeros(N_PSI, dtype=np.float32),               # no face data in AMASS
                gaze=np.array([0.0, 0.0, 1.0], dtype=np.float32),      # neutral forward-gaze placeholder
                delta=np.zeros(N_DELTA, dtype=np.float32),
                cam=np.zeros(N_CAM, dtype=np.float32),                  # AMASS has no camera; 'trans' is
                                                                          # body-relative, not camera translation
                timestamp=i / src_fps,
            )
        )

    subject_id = os.path.splitext(os.path.basename(npz_path))[0]
    return Sequence(
        beta=beta,
        alpha=alpha,
        frames=frames,
        subject_id=subject_id,
        source_dataset="AMASS_CMU",
        frame_rate_hz=target_fps,
    )


def convert_all(raw_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Only the '*_stageii.npz' files are final motion sequences with a 'poses' field.
    # '*_stagei.npz' / 'neutral_stagei.npz' are AMASS's own intermediate calibration fits
    # (shape-only, no motion) and are intentionally excluded here rather than attempted and failed.
    npz_files = sorted(glob.glob(os.path.join(raw_dir, "**", "*_stageii.npz"), recursive=True))
    skipped = len(glob.glob(os.path.join(raw_dir, "**", "*stagei.npz"), recursive=True))
    if skipped:
        print(f"Skipping {skipped} stage-I calibration file(s) (no motion data, expected).")
    if not npz_files:
        print(f"No '*_stageii.npz' motion files found under {raw_dir}. Nothing to convert.")
        return

    n_ok, n_fail = 0, 0
    for path in npz_files:
        try:
            seq = load_amass_npz(path)
            seq.validate()
            out_path = os.path.join(out_dir, os.path.basename(path))
            seq.to_npz(out_path)
            n_ok += 1
            if n_ok % 20 == 0:
                print(f"...converted {n_ok} sequences so far")
        except Exception as e:
            n_fail += 1
            print(f"FAILED: {path} -> {e}")

    print(f"Done. Converted {n_ok} sequences, {n_fail} failed, out of {len(npz_files)} total.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/amass/raw/CMU")
    parser.add_argument("--out_dir", default="data/amass/converted/CMU")
    args = parser.parse_args()
    convert_all(args.raw_dir, args.out_dir)
