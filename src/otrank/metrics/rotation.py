"""PyTorch rotation-equivariance metrics."""

from __future__ import annotations

import torch


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
