"""
Delta-prediction utility.

Predicting the RAW absolute upper-body descriptor at each frame makes the model's job harder than
necessary: frame-to-frame human motion is mostly "close to no change", which is exactly what the
zero-parameter Persistence baseline gets for free. Both A1 and A3 LOST to Persistence at every
rollout horizon when predicting absolute values directly (see scripts/eval_ablations.py results,
2026-08-21 run) -- switching to delta prediction (predict the CHANGE from the previous frame, then
add it back) directly encodes that "usually small change" prior, and is standard practice in human
motion prediction literature for exactly this reason.

Convention used throughout scripts/train_prior.py and scripts/eval_ablations.py after this change:
models still take the ABSOLUTE previous frame as INPUT (z_upper_prev, unchanged) -- only their
mean/log_var OUTPUT is now interpreted as a delta: predicted_absolute = prev_absolute + mean.
"""
import torch


def reconstruct_absolute(prev_absolute: torch.Tensor, predicted_delta: torch.Tensor) -> torch.Tensor:
    """prev_absolute, predicted_delta: same shape -> predicted absolute value."""
    return prev_absolute + predicted_delta
