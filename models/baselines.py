"""
Baseline models for the ablation matrix (review Section 11). These exist purely to answer:
does the proposed face-body coupling (models/coupling_prior.py's ProbabilisticCouplingPrior, A3)
actually help, compared to simpler alternatives?

A0 (fully independent face/body, e.g. separate VIBE+DECA-style pretrained systems) is NOT
implemented here -- it requires integrating external pretrained initializers this project hasn't
set up. Flagged as a known scope gap, not silently skipped.

A1 (this file's UnconditionalUpperBodyPrior): a temporal model for upper-body motion that uses
ONLY the upper body's own history -- no face, gaze, or head conditioning at all. This is the
critical control condition: if the proposed coupling model (A3) doesn't beat this, there's no
evidence the face-body coupling mechanism itself is doing anything useful.

A2 (deterministic coupling) is already implemented as DeterministicCoupling in
models/coupling_prior.py -- reused here, not duplicated.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class UnconditionalUpperBodyPrior(nn.Module):
    """
    A1: predicts p(z_upper_t | z_upper_{t-1}) using ONLY upper-body history -- deliberately no
    z_face_t or gaze_t input. Same architecture family (GRU + diagonal Gaussian head) as the
    proposed ProbabilisticCouplingPrior, so any performance difference between A1 and A3 reflects
    the value of face/gaze conditioning itself, not a difference in model capacity.
    """

    def __init__(
        self,
        upper_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        min_log_var: float = -6.0,
        max_log_var: float = 3.0,
    ):
        super().__init__()
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

        self.gru = nn.GRU(upper_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 2 * upper_dim)

    def forward(
        self, z_upper_prev: torch.Tensor, h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """z_upper_prev: (B, T, upper_dim) -> mean, log_var: (B, T, upper_dim); h_n hidden state."""
        out, h_n = self.gru(z_upper_prev, h0)
        stats = self.head(out)
        mean, log_var = stats.chunk(2, dim=-1)
        log_var = log_var.clamp(self.min_log_var, self.max_log_var)
        return mean, log_var, h_n


if __name__ == "__main__":
    # Smoke test -- same shapes ProbabilisticCouplingPrior uses, minus face/gaze inputs.
    B, T, upper_dim = 2, 30, 24
    model = UnconditionalUpperBodyPrior(upper_dim)
    z_upper_prev = torch.randn(B, T, upper_dim)
    mean, log_var, h_n = model(z_upper_prev)
    print(f"OK: mean {tuple(mean.shape)}, log_var {tuple(log_var.shape)}, hidden {tuple(h_n.shape)}")

    from losses.losses import coupling_nll_loss
    target = torch.randn(B, T, upper_dim)
    nll = coupling_nll_loss(mean, log_var, target)
    print(f"OK: NLL = {nll.item():.4f}")
