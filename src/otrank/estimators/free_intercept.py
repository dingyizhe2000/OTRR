"""Joint-intercept linear baselines for the paper's linear simulation."""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import solve
import torch

from .baselines import LinearModelFit


def free_intercept_cutoff(
    y: torch.Tensor, loss: str, tau_multiplier: float = 1.345,
) -> np.ndarray | float | None:
    if loss == "ols":
        return None
    if not math.isfinite(tau_multiplier) or tau_multiplier <= 0:
        raise ValueError("tau_multiplier must be finite and positive")
    if loss == "cw_huber":
        median = torch.median(y, dim=0).values
        mad = torch.median(torch.abs(y - median), dim=0).values / 0.6745
        return tau_multiplier * np.maximum(mad.detach().cpu().double().numpy(), 1e-6)
    if loss == "radial_huber":
        return tau_multiplier * math.sqrt(y.shape[1])
    raise ValueError(f"unsupported joint-intercept loss: {loss}")


def fit_free_intercept_linear(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    loss: str,
    tau_multiplier: float = 1.345,
    ridge: float = 1e-8,
    max_iter: int = 80,
    tol: float = 1e-7,
) -> LinearModelFit:
    """Fit a shared-design LS/Huber model on the augmented design in float64."""
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or x.device != y.device:
        raise ValueError("x and y must be aligned matrices on the same device")
    if not math.isfinite(ridge) or ridge < 0 or max_iter < 1 or not math.isfinite(tol) or tol <= 0:
        raise ValueError("invalid ridge, iteration cap, or tolerance")
    cutoff = free_intercept_cutoff(y, loss, tau_multiplier)
    response = y.detach().cpu().numpy().astype(np.float64)
    z = np.column_stack((np.ones(len(x)), x.detach().cpu().numpy()))
    penalty = np.eye(z.shape[1]) * ridge
    penalty[0, 0] = 0.0
    theta = solve(z.T @ z + penalty, z.T @ response, assume_a="pos").T

    def irls(beta, target, weights):
        for _ in range(max_iter):
            w = weights(target - z @ beta.T)
            new = solve(z.T @ (z * w[:, None]) + penalty,
                        z.T @ (target * (w if target.ndim == 1 else w[:, None])),
                        assume_a="pos").T
            change = np.linalg.norm(new - beta) / max(1.0, np.linalg.norm(beta))
            if change <= tol:
                return new
            beta = new
        raise RuntimeError(f"{loss} joint-intercept IRLS did not converge in {max_iter} iterations")

    if loss == "cw_huber":
        for j in range(response.shape[1]):
            theta[j] = irls(theta[j], response[:, j], lambda residual: np.clip(
                cutoff[j] / np.maximum(np.abs(residual), 1e-300), 1e-8, 1.0))
    elif loss == "radial_huber":
        theta = irls(theta, response, lambda residual: np.clip(
            cutoff / np.maximum(np.linalg.norm(residual, axis=1), 1e-12), 1e-8, 1.0))

    coef = torch.as_tensor(theta[:, 1:].copy(), device=x.device)
    intercept = torch.as_tensor(theta[:, 0].copy()[None, :], device=x.device)
    return LinearModelFit(coef=coef, x_mean=coef.new_zeros((1, x.shape[1])),
                          y_mean=intercept, method=f"{loss}_free_intercept")
