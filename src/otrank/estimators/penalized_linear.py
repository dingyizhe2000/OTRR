"""L2-penalized linear baselines with a common objective normalization."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .baselines import (
    LinearModelFit,
    fit_coordinatewise_huber,
    fit_ols,
    fit_radial_huber,
)
from .neural_baselines import _IntegratedScoreLoss, estimate_antitonic_score


@dataclass
class LinearASMConfig:
    """Python coordinatewise ASM with an optional Frobenius penalty."""

    l2_penalty: float = 0.0
    alt_iter: int = 1
    max_iter: int = 65
    score_grid_size: int = 3000
    kernel_points: int = 32768
    bandwidth: str | float = "nrd0"
    tolerance_grad: float = 1e-9
    tolerance_change: float = 1e-12


def _ridge_system_penalty(n: int, l2_penalty: float) -> float:
    """Convert average-loss lambda to the sum-loss normal-equation scale."""

    value = float(l2_penalty)
    if value < 0.0:
        raise ValueError("l2_penalty must be nonnegative")
    return max(float(n) * value, 1e-10)


def fit_l2_least_squares(
    x: torch.Tensor,
    y: torch.Tensor,
    l2_penalty: float = 0.0,
) -> LinearModelFit:
    """Fit centered least squares plus ``lambda ||B||_F^2 / 2``."""

    fit = fit_ols(x, y, ridge=_ridge_system_penalty(x.shape[0], l2_penalty))
    fit.method = "l2_least_squares"
    return fit


def fit_l2_coordinatewise_huber(
    x: torch.Tensor,
    y: torch.Tensor,
    l2_penalty: float = 0.0,
    *,
    tau: float = 1.345,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> LinearModelFit:
    """Fit coordinatewise Huber loss with a shared L2 penalty."""

    fit = fit_coordinatewise_huber(
        x,
        y,
        tau=tau,
        max_iter=max_iter,
        tol=tol,
        ridge=_ridge_system_penalty(x.shape[0], l2_penalty),
        intercept_mode="center",
    )
    fit.method = "l2_coordinatewise_huber"
    return fit


def fit_l2_radial_huber(
    x: torch.Tensor,
    y: torch.Tensor,
    l2_penalty: float = 0.0,
    *,
    tau: float | None = None,
    tau_multiplier: float = 1.345,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> LinearModelFit:
    """Fit radial Huber loss with a shared L2 penalty."""

    fit = fit_radial_huber(
        x,
        y,
        tau=tau,
        tau_multiplier=tau_multiplier,
        max_iter=max_iter,
        tol=tol,
        ridge=_ridge_system_penalty(x.shape[0], l2_penalty),
        intercept_mode="center",
    )
    fit.method = "l2_radial_huber"
    return fit


def fit_l2_coordinatewise_asm(
    x: torch.Tensor,
    y: torch.Tensor,
    config: LinearASMConfig | None = None,
) -> LinearModelFit:
    """Fit Python linear ASM by alternating score estimation and convex fitting.

    The pilot is centered ridge regression with the same L2 penalty. At each
    alternation, the antitonic scores are estimated from current residuals and
    then held fixed while minimizing their integrated losses plus the penalty.
    """

    config = config or LinearASMConfig()
    l2_penalty = float(config.l2_penalty)
    if l2_penalty < 0.0:
        raise ValueError("l2_penalty must be nonnegative")
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    xc = x - x_mean
    yc = y - y_mean
    pilot = fit_l2_least_squares(x, y, l2_penalty=l2_penalty)
    coef = pilot.coef.detach().clone()

    for _ in range(max(1, int(config.alt_iter))):
        residuals = yc - xc @ coef.T
        score_losses = [
            _IntegratedScoreLoss(
                estimate_antitonic_score(
                    residuals[:, j],
                    score_grid_size=int(config.score_grid_size),
                    kernel_points=int(config.kernel_points),
                    bandwidth=config.bandwidth,
                ),
                device=x.device,
                dtype=x.dtype,
            )
            for j in range(y.shape[1])
        ]
        parameter = torch.nn.Parameter(coef.clone())
        optimizer = torch.optim.LBFGS(
            [parameter],
            lr=1.0,
            max_iter=max(1, int(config.max_iter)),
            tolerance_grad=float(config.tolerance_grad),
            tolerance_change=float(config.tolerance_change),
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            current_residual = yc - xc @ parameter.T
            data_loss = torch.stack(
                [score_losses[j](current_residual[:, j]) for j in range(y.shape[1])],
                dim=1,
            ).sum(dim=1).mean()
            loss = data_loss + 0.5 * l2_penalty * parameter.square().sum()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite penalized ASM objective")
            loss.backward()
            return loss

        optimizer.step(closure)
        coef = parameter.detach().clone()

    return LinearModelFit(
        coef=coef,
        x_mean=x_mean,
        y_mean=y_mean,
        method="l2_coordinatewise_asm",
    )
