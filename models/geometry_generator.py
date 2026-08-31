"""
Geometry generator: V_t = G(beta, alpha, theta_t, psi_t, delta_t)

Wraps the `smplx` package's SMPL-X model. SMPL-X already includes jaw pose, eye pose, and
facial expression blendshapes natively (unlike original SMPL), so we use SMPL-X's own
expression space as the FLAME-compatible psi_t component rather than loading a separate
FLAME model. This matches the "one canonical generator equation" recommendation in the review.

Requires:
    pip install smplx
    SMPL-X model files downloaded from https://smpl-x.is.tue.mpg.de/ and placed under
    <body_models_path>/smplx/SMPLX_{MALE,FEMALE,NEUTRAL}.npz

Usage:
    gen = GeometryGenerator(body_models_path="./data/body_models", gender="neutral")
    output = gen(beta, alpha_unused, theta, psi, delta_unused)
    output.vertices   # (B, 10475, 3)
    output.joints      # (B, 55, 3)  approx, depends on smplx version
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

try:
    import smplx
except ImportError as e:
    raise ImportError(
        "The 'smplx' package is required. Install with: pip install smplx"
    ) from e


@dataclass
class GeneratorOutput:
    vertices: torch.Tensor      # (B, V, 3) mesh vertices in canonical pose/frame
    joints: torch.Tensor         # (B, J, 3) joint locations
    faces: torch.Tensor          # (F, 3) triangle indices (shared across batch, not batched)


class GeometryGenerator(nn.Module):
    """
    Thin wrapper around smplx.SMPLX that exposes the project's canonical parameter names
    (beta, theta, psi) instead of SMPL-X's native argument names (betas, body_pose, expression, ...).

    theta_t layout expected here (must match preprocessing/canonical_state.py N_THETA):
        theta[0:3]      global_orient (root, axis-angle)
        theta[3:66]      body_pose (21 joints x 3, axis-angle)
        theta[66:69]     jaw_pose (1 joint x 3)
        theta[69:75]     leye_pose + reye_pose (2 joints x 3)
        theta[75:165]    left_hand_pose + right_hand_pose (30 joints x 3)
        -> total 165 dims. NOTE: this differs from the N_THETA=165 placeholder in
           canonical_state.py (55*3=165) -- they already match, but double check joint counts
           against your actual smplx model version (num_pca_comps for hands can shrink this a lot).
    """

    def __init__(
        self,
        body_models_path: str,
        gender: str = "neutral",
        n_betas: int = 10,
        n_expression: int = 50,
        use_pca_hands: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.model = smplx.create(
            model_path=body_models_path,
            model_type="smplx",
            gender=gender,
            num_betas=n_betas,
            num_expression_coeffs=n_expression,
            use_pca=use_pca_hands,
            batch_size=1,  # overridden dynamically per forward call where possible
        ).to(device)
        self.n_betas = n_betas
        self.n_expression = n_expression

    def forward(
        self,
        beta: torch.Tensor,     # (B, n_betas)
        theta: torch.Tensor,     # (B, 165) see layout above
        psi: torch.Tensor,        # (B, n_expression)
        delta: torch.Tensor | None = None,   # (B, N_DELTA) high-freq detail; unused for now
    ) -> GeneratorOutput:
        B = beta.shape[0]

        global_orient = theta[:, 0:3]
        body_pose = theta[:, 3:66]
        jaw_pose = theta[:, 66:69]
        leye_pose = theta[:, 69:72]
        reye_pose = theta[:, 72:75]
        left_hand_pose = theta[:, 75:120]
        right_hand_pose = theta[:, 120:165]

        out = self.model(
            betas=beta,
            global_orient=global_orient,
            body_pose=body_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            expression=psi,
            return_verts=True,
        )

        # delta_t (high-frequency detail, D(delta_t, psi_t) in the review's equation) is not yet
        # implemented -- vertices are the raw SMPL-X output until a detail/displacement branch
        # is added. This is a known simplification, not a bug.
        faces = torch.as_tensor(self.model.faces.astype("int64"), device=beta.device)

        return GeneratorOutput(vertices=out.vertices, joints=out.joints, faces=faces)


if __name__ == "__main__":
    # Smoke test with random parameters and NO real SMPL-X model files -- this will fail until you
    # download the model files, by design. Run this after Step 7 (SMPL-X download) from the setup
    # instructions, with body_models_path pointing at your actual data/body_models/ directory.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--body_models_path", default="./data/body_models")
    args = parser.parse_args()

    gen = GeometryGenerator(body_models_path=args.body_models_path)
    beta = torch.zeros(1, 10)
    theta = torch.zeros(1, 165)
    psi = torch.zeros(1, 50)
    output = gen(beta, theta, psi)
    print(f"OK: vertices {tuple(output.vertices.shape)}, joints {tuple(output.joints.shape)}, "
          f"faces {tuple(output.faces.shape)}")
