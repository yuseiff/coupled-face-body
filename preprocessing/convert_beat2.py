"""
Converter: BEAT2/EMAGE (SMPL-X + FLAME format) -> canonical Sequence format.

Based on the actual on-disk schema of beat_english_v2.0.0/smplxflame_30/*.npz (verified 2026-08-21):
    betas               (300,)       SMPL-X shape components; we keep only the first N_BETA=10
    poses               (T, 165)     = [root_orient(3), pose_body(63), pose_jaw(3), pose_eye(6),
                                         pose_hand(90)] -- matches our theta layout exactly, same as AMASS
    expressions         (T, 100)     FLAME/SMPL-X expression coefficients; we keep the first N_PSI=50
    trans               (T, 3)       body root translation (NOT a camera pose) -- mapped into cam_t as
                                       an approximation; revisit if/when a real camera model is added
    model, gender, mocap_frame_rate  scalars

THIS IS THE ANCHOR DATASET for the coupling experiment: unlike AMASS, it has real synchronized body
pose AND facial expression per frame from the same sequence. Gaze is NOT present in this file format --
left as a neutral placeholder here; add a real gaze estimator pass (e.g. MediaPipe/Gaze360-based) over
the corresponding video/audio-aligned frames later if gaze supervision is required for the coupling prior.

Usage:
    python preprocessing/convert_beat2.py \
        --raw_dir data/beat2/raw/beat_english_v2.0.0/smplxflame_30 \
        --out_dir data/beat2/converted/beat_english_v2.0.0
"""
import argparse
import glob
import os

import numpy as np

from canonical_state import Sequence, FrameState, N_BETA, N_ALPHA, N_PSI, N_GAZE, N_DELTA, N_CAM, N_THETA

TARGET_FPS = 30.0


def load_beat2_npz(npz_path: str, target_fps: float = TARGET_FPS) -> Sequence:
    data = np.load(npz_path, allow_pickle=True)

    poses = data["poses"]                  # (T, 165)
    betas_full = data["betas"]              # (300,)
    expressions_full = data["expressions"]   # (T, 100)
    trans = data["trans"]                     # (T, 3)
    src_fps = float(data["mocap_frame_rate"])
    T = poses.shape[0]

    if poses.shape[1] != N_THETA:
        raise ValueError(f"{npz_path}: expected theta dim {N_THETA}, got {poses.shape[1]}.")
    if expressions_full.shape[0] != T or trans.shape[0] != T:
        raise ValueError(f"{npz_path}: frame count mismatch between poses/expressions/trans.")

    # BEAT2 is already recorded at ~30fps per the report, but resample defensively in case a
    # sub-file differs, using the same stride-subsampling approach as convert_amass.py.
    stride = max(1, round(src_fps / target_fps))
    frame_indices = np.arange(0, T, stride)

    beta = betas_full[:N_BETA].astype(np.float32)
    alpha = np.zeros(N_ALPHA, dtype=np.float32)  # no separate face-identity field in this format;
                                                     # identity is implicitly folded into 'betas' + expressions

    frames = []
    for i in frame_indices:
        theta = poses[i].astype(np.float32)
        psi_full = expressions_full[i].astype(np.float32)
        psi = psi_full[:N_PSI] if psi_full.shape[0] >= N_PSI else np.pad(psi_full, (0, N_PSI - psi_full.shape[0]))
        frames.append(
            FrameState(
                theta=theta,
                psi=psi,
                gaze=np.array([0.0, 0.0, 1.0], dtype=np.float32),  # placeholder -- see module docstring
                delta=np.zeros(N_DELTA, dtype=np.float32),
                cam=trans[i].astype(np.float32),                     # approximation: body root translation,
                                                                        # not a true camera pose (see docstring)
                timestamp=i / src_fps,
            )
        )

    subject_id = os.path.splitext(os.path.basename(npz_path))[0]
    return Sequence(
        beta=beta,
        alpha=alpha,
        frames=frames,
        subject_id=subject_id,
        source_dataset="BEAT2_english",
        frame_rate_hz=target_fps,
    )


def convert_all(raw_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    npz_files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.npz"), recursive=True))
    if not npz_files:
        print(f"No .npz files found under {raw_dir}. Nothing to convert.")
        return

    n_ok, n_fail = 0, 0
    for path in npz_files:
        try:
            seq = load_beat2_npz(path)
            seq.validate()
            out_path = os.path.join(out_dir, os.path.basename(path))
            seq.to_npz(out_path)
            n_ok += 1
            if n_ok % 50 == 0:
                print(f"...converted {n_ok} sequences so far")
        except Exception as e:
            n_fail += 1
            print(f"FAILED: {path} -> {e}")

    print(f"Done. Converted {n_ok} sequences, {n_fail} failed, out of {len(npz_files)} total.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/beat2/raw/beat_english_v2.0.0/smplxflame_30")
    parser.add_argument("--out_dir", default="data/beat2/converted/beat_english_v2.0.0")
    args = parser.parse_args()
    convert_all(args.raw_dir, args.out_dir)
