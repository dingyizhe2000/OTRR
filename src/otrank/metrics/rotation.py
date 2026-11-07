"""PyTorch rotation-equivariance metrics."""

from __future__ import annotations

import torch


def rotation_coefficient_max_error(coef: torch.Tensor, rotated_coef: torch.Tensor, rotation: torch.Tensor) -> float:
    """Entrywise max |Q.T B_Q-B| for d-by-p slopes, excluding intercepts."""
    if coef.ndim != 2 or rotated_coef.shape != coef.shape or rotation.shape != (coef.shape[0], coef.shape[0]):
        raise ValueError("expected equal d-by-p slopes and a d-by-d response rotation")
    arrays = [value.detach().cpu().double() for value in (coef, rotated_coef, rotation)]
    if not all(torch.isfinite(value).all() for value in arrays):
        raise ValueError("coefficient equivariance inputs must be finite")
    b, b_rotated, q = arrays
    return float((q.T @ b_rotated-b).abs().max())


def rotation_prediction_gap(pred: torch.Tensor, rotated_back_pred: torch.Tensor) -> float:
    """Root mean squared gap between predictions before/after response rotation."""

    diff = pred - rotated_back_pred
    return float(torch.sqrt(torch.mean(torch.sum(diff**2, dim=1))).item())


def rotation_prediction_mse_per_dim(pred: torch.Tensor, rotated_back_pred: torch.Tensor) -> float:
    """Mean squared prediction gap per response coordinate after rotation back."""

    return float(torch.mean((pred - rotated_back_pred) ** 2).item())


def rotation_prediction_rmse_per_dim(pred: torch.Tensor, rotated_back_pred: torch.Tensor) -> float:
    """Root mean squared prediction gap per response coordinate after rotation back."""

    return float(torch.sqrt(torch.mean((pred - rotated_back_pred) ** 2)).item())
