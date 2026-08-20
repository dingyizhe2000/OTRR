"""PyTorch baseline estimators for multivariate linear regression."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LinearModelFit:
    coef: torch.Tensor
    x_mean: torch.Tensor
    y_mean: torch.Tensor
    method: str

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean) @ self.coef.T + self.y_mean


def _center_xy(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    return x - x_mean, y - y_mean, x_mean, y_mean


def fit_ols(x: torch.Tensor, y: torch.Tensor, ridge: float = 1e-8) -> LinearModelFit:
    """Fit centered multivariate least squares with a small ridge stabilizer."""

    xc, yc, x_mean, y_mean = _center_xy(x, y)
    xtx = xc.T @ xc
    penalty = ridge * torch.eye(xtx.shape[0], device=x.device, dtype=x.dtype)
    coef_t = torch.linalg.solve(xtx + penalty, xc.T @ yc)
    return LinearModelFit(coef=coef_t.T, x_mean=x_mean, y_mean=y_mean, method="ols")


def fit_zero_intercept_ols(x: torch.Tensor, y: torch.Tensor, ridge: float = 1e-8) -> LinearModelFit:
    """Fit least squares for ``Y = X B^T + error`` without centering or intercept."""

    xtx = x.T @ x
    penalty = ridge * torch.eye(xtx.shape[0], device=x.device, dtype=x.dtype)
    coef_t = torch.linalg.solve(xtx + penalty, x.T @ y)
    return LinearModelFit(
        coef=coef_t.T,
        x_mean=torch.zeros(1, x.shape[1], device=x.device, dtype=x.dtype),
        y_mean=torch.zeros(1, y.shape[1], device=y.device, dtype=y.dtype),
        method="zero_intercept_ols",
    )


def _fit_weighted_ls(x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, ridge: float) -> torch.Tensor:
    weights = torch.clamp(weights, min=1e-8)
    sqrt_w = torch.sqrt(weights)
    xw = x * sqrt_w[:, None]
    yw = y * sqrt_w if y.ndim == 1 else y * sqrt_w[:, None]
    xtx = xw.T @ xw
    return torch.linalg.solve(xtx + ridge * torch.eye(xtx.shape[0], device=x.device, dtype=x.dtype), xw.T @ yw)


def fit_coordinatewise_huber(
    x: torch.Tensor,
    y: torch.Tensor,
    tau: float = 1.345,
    max_iter: int = 100,
    tol: float = 1e-7,
    ridge: float = 1e-8,
    intercept_mode: str = "zero",
) -> LinearModelFit:
    """Fit independent Huber regressions for each response coordinate.

    The implementation uses iteratively reweighted least squares. By default it
    fits the zero-intercept model used in the current simulations.
    """

    mode = intercept_mode.lower()
    if mode == "zero":
        xc = x
        yc = y
        x_mean = torch.zeros(1, x.shape[1], device=x.device, dtype=x.dtype)
        y_mean = torch.zeros(1, y.shape[1], device=y.device, dtype=y.dtype)
        ols = fit_zero_intercept_ols(x, y, ridge=ridge)
    elif mode in {"center", "intercept"}:
        xc, yc, x_mean, y_mean = _center_xy(x, y)
        ols = fit_ols(x, y, ridge=ridge)
    else:
        raise ValueError("intercept_mode must be 'zero', 'center', or 'intercept'")
    d = yc.shape[1]
    coef = torch.zeros(d, xc.shape[1], device=x.device, dtype=x.dtype)
    coef[:] = ols.coef

    for j in range(d):
        beta = coef[j].clone()
        response = yc[:, j]
        median = torch.median(response)
        scale = torch.median(torch.abs(response - median)) / 0.6745
        scale = max(float(scale.item()), 1e-6)
        cutoff = tau * scale
        for _ in range(max_iter):
            residual = response - xc @ beta
            abs_r = torch.abs(residual)
            weights = torch.ones_like(abs_r)
            large = abs_r > cutoff
            weights[large] = cutoff / abs_r[large]
            beta_new = _fit_weighted_ls(xc, response, weights, ridge=ridge)
            denom = max(1.0, float(torch.linalg.norm(beta).item()))
            if torch.linalg.norm(beta_new - beta) <= tol * denom:
                beta = beta_new
                break
            beta = beta_new
        coef[j] = beta

    return LinearModelFit(coef=coef, x_mean=x_mean, y_mean=y_mean, method="coordinatewise_huber")


def fit_radial_huber(
    x: torch.Tensor,
    y: torch.Tensor,
    tau: float | None = None,
    tau_multiplier: float = 1.345,
    max_iter: int = 100,
    tol: float = 1e-7,
    ridge: float = 1e-8,
    intercept_mode: str = "zero",
    eps: float = 1e-12,
) -> LinearModelFit:
    """Fit multivariate linear regression with radial Huber loss.

    The objective is ``sum_i h_tau(||y_i - B x_i||_2)``.  The IRLS weights are
    ``min(1, tau / ||r_i||_2)``, which corresponds to the score
    ``r_i`` inside the ball and ``tau r_i / ||r_i||_2`` outside the ball.
    By default ``tau = 1.345 * sqrt(d)`` for ``d`` response coordinates.
    """

    mode = intercept_mode.lower()
    if mode == "zero":
        xc = x
        yc = y
        x_mean = torch.zeros(1, x.shape[1], device=x.device, dtype=x.dtype)
        y_mean = torch.zeros(1, y.shape[1], device=y.device, dtype=y.dtype)
        fit = fit_zero_intercept_ols(x, y, ridge=ridge)
    elif mode in {"center", "intercept"}:
        xc, yc, x_mean, y_mean = _center_xy(x, y)
        fit = fit_ols(x, y, ridge=ridge)
    else:
        raise ValueError("intercept_mode must be 'zero', 'center', or 'intercept'")

    d = yc.shape[1]
    cutoff = float(tau) if tau is not None else float(tau_multiplier) * (float(d) ** 0.5)
    if cutoff <= 0:
        raise ValueError("radial Huber tau must be positive")

    coef_t = fit.coef.T.clone()
    for _ in range(max_iter):
        residual = yc - xc @ coef_t
        norm = torch.linalg.norm(residual, dim=1)
        safe_norm = torch.clamp(norm, min=eps)
        weights = torch.ones_like(norm)
        large = norm > cutoff
        weights[large] = cutoff / safe_norm[large]
        coef_t_new = _fit_weighted_ls(xc, yc, weights, ridge=ridge)
        denom = max(1.0, float(torch.linalg.norm(coef_t).item()))
        if torch.linalg.norm(coef_t_new - coef_t) <= tol * denom:
            coef_t = coef_t_new
            break
        coef_t = coef_t_new

    return LinearModelFit(coef=coef_t.T, x_mean=x_mean, y_mean=y_mean, method="radial_huber")
