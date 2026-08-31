"""
Individual loss terms from the training objective (Section 8 of the implementation report):

L = sum_t [ lambda_key*L_key + lambda_face*L_face + lambda_gaze*L_gaze + lambda_id*L_id
          + lambda_expr*L_expr + lambda_sil*L_sil + lambda_photo*L_photo ]
    + lambda_temp*L_temp + lambda_coll*L_coll + lambda_prior*L_prior + lambda_couple*L_couple

Each function returns a scalar loss for a batch. Combine them in optimization/differentiable_refinement.py
using the weights from configs/default.yaml.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def huber(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Elementwise Huber (smooth L1), used instead of raw L1/L2 for robustness to outliers/mis-detections."""
    abs_x = x.abs()
    quad = torch.minimum(abs_x, torch.full_like(abs_x, delta))
    lin = abs_x - quad
    return 0.5 * quad**2 + delta * lin


def keypoint_loss(pred_2d: torch.Tensor, obs_2d: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
    """
    L_key: robust reprojection loss on 2D body keypoints, weighted by detector confidence.
    pred_2d, obs_2d: (B, T, K, 2)   confidence: (B, T, K)
    """
    diff = huber(pred_2d - obs_2d).sum(dim=-1)  # (B, T, K)
    return (diff * confidence).sum() / confidence.sum().clamp_min(1e-6)


def face_landmark_loss(
    pred_2d: torch.Tensor, obs_2d: torch.Tensor, confidence: torch.Tensor,
    eye_mouth_indices: list[int] | None = None, region_weight: float = 2.0,
) -> torch.Tensor:
    """
    L_face: normalized face landmark distance, with extra weight around eyes/mouth per the report.
    pred_2d, obs_2d: (B, T, L, 2)   confidence: (B, T, L)
    """
    diff = huber(pred_2d - obs_2d).sum(dim=-1)  # (B, T, L)
    weights = confidence.clone()
    if eye_mouth_indices:
        weights[..., eye_mouth_indices] *= region_weight
    return (diff * weights).sum() / weights.sum().clamp_min(1e-6)


def gaze_loss(pred_gaze: torch.Tensor, obs_gaze: torch.Tensor) -> torch.Tensor:
    """
    L_gaze: angular distance between predicted and observed unit gaze vectors.
    pred_gaze, obs_gaze: (B, T, 3), assumed unit-normalized.
    """
    cos_sim = F.cosine_similarity(pred_gaze, obs_gaze, dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
    angle = torch.acos(cos_sim)  # radians
    return angle.mean()


def identity_loss(pred_embedding: torch.Tensor, ref_embedding: torch.Tensor) -> torch.Tensor:
    """
    L_id: cosine distance between identity embeddings of input and rendered/reconstructed face.
    Embeddings should come from a fixed pretrained face-recognition network (e.g. ArcFace) —
    that network is NOT implemented here; this function only consumes its output.
    pred_embedding, ref_embedding: (B, T, D)
    """
    cos_sim = F.cosine_similarity(pred_embedding, ref_embedding, dim=-1)
    return (1.0 - cos_sim).mean()


def expression_consistency_loss(pred_psi: torch.Tensor, obs_expr_features: torch.Tensor) -> torch.Tensor:
    """
    L_expr: distance between predicted expression parameters and observed expression/AU descriptors.
    Placeholder assumes obs_expr_features is already projected into the same space as pred_psi;
    if using discrete AU labels instead, replace with a classification loss (BCE) per AU.
    pred_psi, obs_expr_features: (B, T, N_PSI)
    """
    return F.mse_loss(pred_psi, obs_expr_features)


def temporal_smoothness_loss(theta: torch.Tensor, psi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    L_temp: velocity + acceleration regularization on pose and expression sequences.
    Returns (smoothness_loss, jerk) — report jerk separately as a diagnostic per the report.
    theta, psi: (B, T, D)
    """
    def vel_acc_jerk(x):
        vel = x[:, 1:] - x[:, :-1]
        acc = vel[:, 1:] - vel[:, :-1]
        jerk = acc[:, 1:] - acc[:, :-1]
        return vel, acc, jerk

    theta_vel, theta_acc, theta_jerk = vel_acc_jerk(theta)
    psi_vel, psi_acc, psi_jerk = vel_acc_jerk(psi)

    smooth = theta_vel.pow(2).mean() + theta_acc.pow(2).mean() + psi_vel.pow(2).mean() + psi_acc.pow(2).mean()
    jerk = theta_jerk.abs().mean() + psi_jerk.abs().mean()
    return smooth, jerk


def pose_prior_loss(theta: torch.Tensor, prior_model) -> torch.Tensor:
    """
    L_prior: plausibility prior on body pose (VPoser/DPoser-X-style).
    prior_model must expose a `.log_prob(theta) -> (B, T)` method (e.g. a pretrained VAE prior);
    NOT implemented here — plug in an existing pretrained prior rather than training one from scratch.
    """
    if prior_model is None:
        return torch.tensor(0.0, device=theta.device)
    return -prior_model.log_prob(theta).mean()


def collision_loss(vertices: torch.Tensor, penetration_fn) -> torch.Tensor:
    """
    L_coll: self-intersection/penetration penalty. Off by default in the first baseline
    (lambda_coll = 0 in configs/default.yaml); enable for the stress-test stage (Hi4D).
    penetration_fn must be a callable that returns per-vertex penetration depth given a mesh;
    typically implemented with a signed-distance or BVH-based collision checker (not included here).
    """
    if penetration_fn is None:
        return torch.tensor(0.0, device=vertices.device)
    depth = penetration_fn(vertices)  # (B, T, V) or similar
    return depth.clamp_min(0).pow(2).mean()


def coupling_nll_loss(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    L_couple: negative log-likelihood of a diagonal Gaussian coupling prior
    p_eta(z_upper_t | z_face_t, g_t, z_upper_{t-1}, z_face_{t-1}).
    mean, log_var, target: (B, T, D)  -- output of models/coupling_prior.py
    """
    var = log_var.exp()
    nll = 0.5 * (log_var + (target - mean).pow(2) / var + torch.log(torch.tensor(2 * torch.pi)))
    return nll.sum(dim=-1).mean()
