"""
Descriptor extraction: theta_t, psi_t -> z_face_t, z_upper_t for the coupling prior.

The coupling prior (models/coupling_prior.py) doesn't consume raw theta/psi directly -- it needs
two focused descriptor vectors, per the review's formulation:
    z_face_t  : expression, gaze, and head descriptors  (gaze is handled as a separate arg, see below)
    z_upper_t : shoulder, clavicle, torso, and arm descriptors

This module defines exactly which slice of theta/psi feeds each one, based on the SMPL-X body_pose
joint ordering (21 joints, standard order used throughout this project's geometry_generator.py).

theta layout (165 dims total), as used by models/geometry_generator.py:
    theta[0:3]     global_orient
    theta[3:66]    body_pose        (21 joints x 3, axis-angle)
    theta[66:69]   jaw_pose
    theta[69:72]   leye_pose
    theta[72:75]   reye_pose
    theta[75:120]  left_hand_pose
    theta[120:165] right_hand_pose

SMPL-X body_pose joint order (1-indexed, joint 0 = pelvis/root is NOT in body_pose):
    1 left_hip    4 left_knee   7 left_ankle   10 left_foot   13 left_collar   16 left_shoulder   18 left_elbow   20 left_wrist
    2 right_hip   5 right_knee  8 right_ankle  11 right_foot  14 right_collar  17 right_shoulder  19 right_elbow  21 right_wrist
    3 spine1      6 spine2      9 spine3       12 neck        15 head
"""
from __future__ import annotations

import torch

# --- joint index helpers -------------------------------------------------------------------
def _joint_slice(joint_num_1indexed: int) -> slice:
    """theta-array slice for a given 1-indexed body_pose joint (see ordering in the docstring)."""
    offset_in_body_pose = (joint_num_1indexed - 1) * 3
    start = 3 + offset_in_body_pose  # +3 to skip global_orient
    return slice(start, start + 3)


NECK = _joint_slice(12)
LEFT_COLLAR = _joint_slice(13)
RIGHT_COLLAR = _joint_slice(14)
HEAD = _joint_slice(15)
LEFT_SHOULDER = _joint_slice(16)
RIGHT_SHOULDER = _joint_slice(17)
LEFT_ELBOW = _joint_slice(18)
RIGHT_ELBOW = _joint_slice(19)
SPINE3 = _joint_slice(9)

JAW = slice(66, 69)
LEYE = slice(69, 72)
REYE = slice(72, 75)

# Resulting descriptor dimensionalities (used to configure models/coupling_prior.py)
N_PSI = 50
FACE_DIM = N_PSI + 3 + 3 + 3 + 3          # psi + jaw + leye + reye + head_joint = 62
UPPER_DIM = 3 * 8                           # neck, l/r collar, l/r shoulder, l/r elbow, spine3 = 24


def extract_face_descriptor(theta: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """
    theta: (..., 165)   psi: (..., N_PSI)   ->   z_face: (..., FACE_DIM)
    Gaze is NOT included here -- the coupling prior takes gaze as a separate argument
    (see models/coupling_prior.py's forward signature), matching the review's z_face/g_t split.
    """
    parts = [psi, theta[..., JAW], theta[..., LEYE], theta[..., REYE], theta[..., HEAD]]
    return torch.cat(parts, dim=-1)


def extract_upper_descriptor(theta: torch.Tensor) -> torch.Tensor:
    """theta: (..., 165) -> z_upper: (..., UPPER_DIM)"""
    parts = [
        theta[..., NECK], theta[..., LEFT_COLLAR], theta[..., RIGHT_COLLAR],
        theta[..., LEFT_SHOULDER], theta[..., RIGHT_SHOULDER],
        theta[..., LEFT_ELBOW], theta[..., RIGHT_ELBOW], theta[..., SPINE3],
    ]
    return torch.cat(parts, dim=-1)


if __name__ == "__main__":
    # Smoke test with random tensors -- no dataset needed.
    B, T = 2, 60
    theta = torch.randn(B, T, 165)
    psi = torch.randn(B, T, N_PSI)

    z_face = extract_face_descriptor(theta, psi)
    z_upper = extract_upper_descriptor(theta)

    assert z_face.shape == (B, T, FACE_DIM), z_face.shape
    assert z_upper.shape == (B, T, UPPER_DIM), z_upper.shape
    print(f"OK: z_face {tuple(z_face.shape)} (FACE_DIM={FACE_DIM}), "
          f"z_upper {tuple(z_upper.shape)} (UPPER_DIM={UPPER_DIM})")
