"""
Canonical sequence representation shared by every dataset converter.

Every dataset (BEAT2, AMASS, BEDLAM, VOCASET, Gaze360, 3DPW, Human3.6M, ...) must be converted
into this single representation before it touches the model. This is the most important
engineering step in the whole project: if datasets disagree on skeleton layout, face parameters,
coordinate frames, or frame rate, the coupling experiment is not comparable across data sources.

Per-frame state (matches Section 6 of the implementation report):
    x_t = [beta, alpha, theta_t, psi_t, g_t, delta_t, c_t]

    beta      body identity/shape params      static over a sequence
    alpha     face identity params             static over a sequence
    theta_t   body/neck/jaw/eye/hand pose       varies per frame
    psi_t     facial expression params          varies per frame
    g_t       gaze direction / latent           varies per frame
    delta_t   optional high-frequency detail    varies per frame
    c_t       camera/scene variables            varies per frame
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# ---- dimensionality conventions (adjust once you commit to exact param counts) ----
N_BETA = 10          # SMPL-X body shape coefficients
N_ALPHA = 100         # FLAME face identity coefficients
N_THETA = 55 * 3      # body(21) + jaw(1) + eyes(2) + hands(30) joints, axis-angle, adjust to your rig
N_PSI = 50            # FLAME expression coefficients
N_GAZE = 3            # unit gaze direction vector
N_DELTA = 0           # high-frequency detail; 0 until you add a detail branch
N_CAM = 3             # translation (tx, ty, tz); extend if optimizing focal length too


@dataclass
class FrameState:
    """theta_t, psi_t, g_t, delta_t, c_t — everything that can vary per frame."""
    theta: np.ndarray   # (N_THETA,)
    psi: np.ndarray      # (N_PSI,)
    gaze: np.ndarray     # (N_GAZE,) unit vector
    delta: np.ndarray    # (N_DELTA,) may be empty
    cam: np.ndarray       # (N_CAM,)
    timestamp: float                    # original timestamp in seconds, preserved for eval
    confidence: Optional[np.ndarray] = None  # per-observation confidence, if available

    def __post_init__(self):
        assert self.theta.shape == (N_THETA,), self.theta.shape
        assert self.psi.shape == (N_PSI,), self.psi.shape
        assert self.gaze.shape == (N_GAZE,), self.gaze.shape
        assert self.cam.shape == (N_CAM,), self.cam.shape
        # normalize gaze to unit vector
        norm = np.linalg.norm(self.gaze)
        if norm > 1e-8:
            self.gaze = self.gaze / norm


@dataclass
class Sequence:
    """A full canonical sequence: static identity params + a list of per-frame states."""
    beta: np.ndarray                # (N_BETA,) static body shape
    alpha: np.ndarray                # (N_ALPHA,) static face identity
    frames: list[FrameState] = field(default_factory=list)
    subject_id: str = ""
    source_dataset: str = ""
    frame_rate_hz: float = 30.0
    coordinate_convention: str = "y_up_camera_forward_z"

    def __post_init__(self):
        assert self.beta.shape == (N_BETA,), self.beta.shape
        assert self.alpha.shape == (N_ALPHA,), self.alpha.shape

    def __len__(self) -> int:
        return len(self.frames)

    def validate(self) -> None:
        """Sanity checks every converter should run before writing a sequence to disk."""
        if len(self.frames) == 0:
            raise ValueError(f"Sequence {self.subject_id} has zero frames.")
        ts = [f.timestamp for f in self.frames]
        if any(t2 <= t1 for t1, t2 in zip(ts, ts[1:])):
            raise ValueError(f"Sequence {self.subject_id} has non-monotonic timestamps.")
        if not self.subject_id:
            raise ValueError("subject_id must be set (needed to prevent identity leakage across splits).")
        if not self.source_dataset:
            raise ValueError("source_dataset must be set for provenance/licensing tracking.")

    def to_npz(self, path: str) -> None:
        self.validate()
        np.savez_compressed(
            path,
            beta=self.beta,
            alpha=self.alpha,
            theta=np.stack([f.theta for f in self.frames]),
            psi=np.stack([f.psi for f in self.frames]),
            gaze=np.stack([f.gaze for f in self.frames]),
            cam=np.stack([f.cam for f in self.frames]),
            timestamps=np.array([f.timestamp for f in self.frames]),
            subject_id=self.subject_id,
            source_dataset=self.source_dataset,
            frame_rate_hz=self.frame_rate_hz,
        )

    @staticmethod
    def dummy(n_frames: int = 10, subject_id: str = "dummy_subj", source_dataset: str = "synthetic") -> "Sequence":
        """Build a synthetic sequence for smoke-testing the pipeline before real data is available."""
        rng = np.random.default_rng(0)
        frames = [
            FrameState(
                theta=rng.normal(size=N_THETA).astype(np.float32) * 0.1,
                psi=rng.normal(size=N_PSI).astype(np.float32) * 0.1,
                gaze=rng.normal(size=N_GAZE).astype(np.float32),
                delta=np.zeros(N_DELTA, dtype=np.float32),
                cam=np.array([0.0, 0.0, 3.0], dtype=np.float32),
                timestamp=i / 30.0,
            )
            for i in range(n_frames)
        ]
        return Sequence(
            beta=rng.normal(size=N_BETA).astype(np.float32) * 0.1,
            alpha=rng.normal(size=N_ALPHA).astype(np.float32) * 0.1,
            frames=frames,
            subject_id=subject_id,
            source_dataset=source_dataset,
        )


if __name__ == "__main__":
    # Smoke test — run this once your environment is set up, before any real dataset is available.
    seq = Sequence.dummy(n_frames=5)
    seq.validate()
    print(f"OK: built dummy sequence with {len(seq)} frames from {seq.source_dataset}")
    seq.to_npz("/tmp/dummy_sequence.npz")
    print("OK: wrote /tmp/dummy_sequence.npz")
