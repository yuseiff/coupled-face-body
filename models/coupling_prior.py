"""
The proposed coupling prior — the actual novel contribution of the paper (Section 7.3 of the
implementation report).

Learns p_eta(z_upper_t | z_face_t, g_t, z_upper_{t-1}, z_face_{t-1}) as a diagonal Gaussian,
instead of a fixed deterministic penalty ||z_upper_t - A*z_face_t||^2, which over-constrains
individual variation.

Also includes the two baselines it must be compared against (Section 11, ablation A0-A3):
    - IndependentBaseline: no coupling at all (A0)
    - DeterministicCoupling: linear/MLP point estimate, no uncertainty (A2, sanity-check baseline)
    - ProbabilisticCouplingPrior: the proposed model (A3)

The critical comparison for the paper is A1 (shared geometry, no coupling) vs A3 (this model).
If A3 doesn't beat A1, the paper doesn't yet have evidence the coupling mechanism helps.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DeterministicCoupling(nn.Module):
    """
    A2 baseline: simple MLP point-estimate mapping from face/gaze descriptors to upper-body
    descriptors, no history, no uncertainty. Exists to test whether a naive coupling helps or
    over-constrains motion, per the review's critique of the original ||z_upper - A*z_face||^2 form.
    """

    def __init__(self, face_dim: int, gaze_dim: int, upper_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(face_dim + gaze_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, upper_dim),
        )

    def forward(self, z_face_t: torch.Tensor, g_t: torch.Tensor) -> torch.Tensor:
        """z_face_t: (B, T, face_dim), g_t: (B, T, gaze_dim) -> (B, T, upper_dim) point estimate."""
        x = torch.cat([z_face_t, g_t], dim=-1)
        return self.net(x)


class ProbabilisticCouplingPrior(nn.Module):
    """
    A3 (proposed model): a GRU over [z_face_t, g_t, z_upper_{t-1}] history, outputting a diagonal
    Gaussian (mean, log_var) over z_upper_t. Trained with the NLL loss in losses/losses.py
    (coupling_nll_loss). Swap the GRU for a TCN or Transformer per configs/default.yaml's
    `coupling_prior_arch` if you want to run that ablation later — interface stays the same.
    """

    def __init__(
        self,
        face_dim: int,
        gaze_dim: int,
        upper_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        min_log_var: float = -6.0,
        max_log_var: float = 3.0,
    ):
        super().__init__()
        self.upper_dim = upper_dim
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

        input_dim = face_dim + gaze_dim + upper_dim  # z_face_t, g_t, z_upper_{t-1} concatenated
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 2 * upper_dim)  # mean + log_var

    def forward(
        self, z_face: torch.Tensor, gaze: torch.Tensor, z_upper_prev: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z_face: (B, T, face_dim) current-frame face/gaze/head descriptors
        gaze: (B, T, gaze_dim)
        z_upper_prev: (B, T, upper_dim) upper-body descriptors shifted by one frame (teacher-forced
                      during training; use the model's own previous prediction at inference time)
        Returns: mean (B, T, upper_dim), log_var (B, T, upper_dim), final_hidden_state
        """
        x = torch.cat([z_face, gaze, z_upper_prev], dim=-1)
        out, h_n = self.gru(x, h0)
        stats = self.head(out)
        mean, log_var = stats.chunk(2, dim=-1)
        log_var = log_var.clamp(self.min_log_var, self.max_log_var)
        return mean, log_var, h_n

    @torch.no_grad()
    def sample(
        self, z_face: torch.Tensor, gaze: torch.Tensor, z_upper_prev: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draw a sample from the predicted distribution — for generation mode, not estimation mode."""
        mean, log_var, _ = self.forward(z_face, gaze, z_upper_prev, h0)
        std = (0.5 * log_var).exp()
        return mean + std * torch.randn_like(mean)


if __name__ == "__main__":
    # Smoke test with random tensors — no dataset or SMPL-X model files required.
    B, T = 2, 30
    face_dim, gaze_dim, upper_dim = 56, 3, 24  # placeholder dims; align with your actual descriptor sizes

    model = ProbabilisticCouplingPrior(face_dim, gaze_dim, upper_dim)
    z_face = torch.randn(B, T, face_dim)
    g = torch.randn(B, T, gaze_dim)
    z_upper_prev = torch.randn(B, T, upper_dim)

    mean, log_var, h_n = model(z_face, g, z_upper_prev)
    print(f"OK: mean {tuple(mean.shape)}, log_var {tuple(log_var.shape)}, hidden {tuple(h_n.shape)}")

    from losses.losses import coupling_nll_loss  # run from repo root for this import to resolve
    target = torch.randn(B, T, upper_dim)
    nll = coupling_nll_loss(mean, log_var, target)
    print(f"OK: coupling NLL = {nll.item():.4f}")
