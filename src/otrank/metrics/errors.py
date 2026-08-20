"""PyTorch error metrics."""

from __future__ import annotations

import torch


def prediction_mse(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.mean(torch.sum((pred - truth) ** 2, dim=1)).item())


def prediction_mse_per_dim(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.mean((pred - truth) ** 2).item())


def prediction_rmse_per_dim(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - truth) ** 2)).item())


def prediction_mae(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - truth)).item())


def translation_adjusted_mse(pred: torch.Tensor, truth: torch.Tensor) -> float:
    """Prediction MSE after removing the best constant response shift."""

    diff = pred - truth
    centered_diff = diff - diff.mean(dim=0, keepdim=True)
    return float(torch.mean(torch.sum(centered_diff**2, dim=1)).item())


def centered_prediction_mse(pred: torch.Tensor, truth: torch.Tensor) -> float:
    """MSE between centered prediction and centered truth."""

    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    truth_centered = truth - truth.mean(dim=0, keepdim=True)
    return prediction_mse(pred_centered, truth_centered)


def coefficient_error(coef: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.linalg.norm(coef - truth, ord="fro").item())


def mean_bias(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float(torch.linalg.norm(torch.mean(pred - truth, dim=0)).item())
