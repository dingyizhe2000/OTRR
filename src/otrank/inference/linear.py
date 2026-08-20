"""Influence functions for the linear baselines used in the paper."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from otrank.estimators.baselines import LinearModelFit

from .covariance import influence_covariance
from otrank.vectorization import COEFFICIENT_VECTORIZATION, vec_coefficient_batch


@dataclass(frozen=True)
class LinearInfluenceResult:
    """Observation-level linear-model influence and diagnostics."""

    influence: np.ndarray
    covariance: np.ndarray
    full_influence: np.ndarray
    full_covariance: np.ndarray
    diagnostics: dict[str, Any]


def _numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _score_and_derivative(
    residual: np.ndarray,
    y: np.ndarray,
    method: str,
    *,
    cw_cutoff: float | np.ndarray | None,
    radial_cutoff: float | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n, d = residual.shape
    label = method.lower().replace("-", "_")
    if label in {"ols", "least_squares", "zero_intercept_ols"}:
        derivative = np.broadcast_to(np.eye(d), (n, d, d)).copy()
        return residual.copy(), derivative, {"score": "least_squares"}

    if label in {"cw_huber", "coordinatewise_huber"}:
        if cw_cutoff is None:
            center = np.median(y, axis=0)
            scale = np.median(np.abs(y - center), axis=0) / 0.6745
            cutoff = 1.345 * np.maximum(scale, 1e-6)
        else:
            cutoff = np.broadcast_to(
                np.asarray(cw_cutoff, dtype=np.float64), (d,)
            ).copy()
        if np.any(cutoff <= 0.0):
            raise ValueError("coordinatewise Huber cutoffs must be positive")
        score = np.clip(residual, -cutoff[None, :], cutoff[None, :])
        derivative = np.zeros((n, d, d), dtype=np.float64)
        index = np.arange(d)
        derivative[:, index, index] = (
            np.abs(residual) < cutoff[None, :]
        ).astype(np.float64)
        return score, derivative, {
            "score": "coordinatewise_huber",
            "cutoffs": cutoff.tolist(),
        }

    if label in {"radial_huber", "multivariate_huber"}:
        cutoff = (
            float(radial_cutoff)
            if radial_cutoff is not None
            else 1.345 * math.sqrt(d)
        )
        if cutoff <= 0.0:
            raise ValueError("radial Huber cutoff must be positive")
        norm = np.linalg.norm(residual, axis=1)
        safe = np.maximum(norm, 1e-12)
        score = residual * np.minimum(1.0, cutoff / safe)[:, None]
        derivative = np.broadcast_to(np.eye(d), (n, d, d)).copy()
        outside = norm > cutoff
        if np.any(outside):
            unit = residual[outside] / safe[outside, None]
            derivative[outside] = cutoff / safe[outside, None, None] * (
                np.eye(d)[None, :, :]
                - unit[:, :, None] * unit[:, None, :]
            )
        return score, derivative, {
            "score": "radial_huber",
            "cutoff": cutoff,
        }
    raise ValueError(f"unsupported linear inference method: {method}")


def linear_model_influence(
    x: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    fit: LinearModelFit,
    *,
    method: str,
    intercept_mode: str = "zero",
    penalty: float = 0.0,
    cw_cutoff: float | np.ndarray | None = None,
    radial_cutoff: float | None = None,
) -> LinearInfluenceResult:
    """Compute a sandwich IF matching the paper's linear estimating equations.

    ``intercept_mode='zero'`` returns the standard column-major ``p*d`` slope
    IF used in the simulations. ``intercept_mode='mean'`` uses a mean-residual equation
    for the intercept and a robust centered-predictor equation for slopes,
    matching the real-data fitters.
    """

    x_np = _numpy(x)
    y_np = _numpy(y)
    coef = _numpy(fit.coef)
    if x_np.ndim != 2 or y_np.ndim != 2:
        raise ValueError("x and y must be matrices")
    if coef.shape != (y_np.shape[1], x_np.shape[1]):
        raise ValueError("fit coefficient shape does not match x and y")
    if penalty < 0.0:
        raise ValueError("penalty must be nonnegative")
    x_mean = _numpy(fit.x_mean).reshape(1, x_np.shape[1])
    y_mean = _numpy(fit.y_mean).reshape(1, y_np.shape[1])
    prediction = (x_np - x_mean) @ coef.T + y_mean
    residual = y_np - prediction
    score, derivative, details = _score_and_derivative(
        residual,
        y_np,
        method,
        cw_cutoff=cw_cutoff,
        radial_cutoff=radial_cutoff,
    )
    n, p = x_np.shape
    d = y_np.shape[1]
    mode = intercept_mode.lower()

    if mode == "zero":
        equation = vec_coefficient_batch(
            np.einsum("nd,np->ndp", score, x_np, optimize=True)
        )
        bread4 = np.einsum(
            "ijk,ia,ib->ajbk",
            derivative,
            x_np,
            x_np,
            optimize=True,
        ) / n
        bread = bread4.reshape(d * p, d * p)
        if penalty:
            bread += float(penalty) * np.eye(d * p)
        full_equation = equation
        slope_indices = np.arange(d * p)
    elif mode in {"mean", "center", "intercept"}:
        q = p + 1
        z = np.concatenate([np.ones((n, 1)), x_np], axis=1)
        x_mean = _numpy(fit.x_mean).reshape(1, p)
        xc = x_np - x_mean
        equations = np.empty((n, d, q), dtype=np.float64)
        equations[:, :, 0] = residual
        equations[:, :, 1:] = score[:, :, None] * xc[:, None, :]
        full_equation = vec_coefficient_batch(equations)
        bread4 = np.zeros((q, d, q, d), dtype=np.float64)
        bread4[0, :, :, :] = np.einsum(
            "jk,b->jbk",
            np.eye(d),
            z.mean(axis=0),
            optimize=True,
        )
        bread4[1:, :, :, :] = np.einsum(
            "ijk,ia,ib->ajbk",
            derivative,
            xc,
            z,
            optimize=True,
        ) / n
        bread = bread4.reshape(d * q, d * q)
        if penalty:
            mask = np.ones(q, dtype=np.float64)
            mask[0] = 0.0
            bread += float(penalty) * np.kron(np.diag(mask), np.eye(d))
        slope_indices = np.arange(d, d * q)
    else:
        raise ValueError("intercept_mode must be 'zero' or 'mean'")

    centered_equation = full_equation - full_equation.mean(axis=0, keepdims=True)
    full_influence = np.linalg.solve(bread, centered_equation.T).T
    influence = full_influence[:, slope_indices]
    full_covariance = influence_covariance(full_influence, center=False)
    covariance = influence_covariance(influence, center=False)
    diagnostics = {
        **details,
        "n": n,
        "predictor_dimension": p,
        "response_dimension": d,
        "parameter_dimension": d * p,
        "full_parameter_dimension": full_influence.shape[1],
        "intercept_mode": mode,
        "penalty": float(penalty),
        "bread_condition_number": float(np.linalg.cond(bread)),
        "estimating_equation_mean_norm": float(
            np.linalg.norm(full_equation.mean(axis=0))
        ),
        "influence_mean_norm": float(np.linalg.norm(influence.mean(axis=0))),
        "vectorization": COEFFICIENT_VECTORIZATION,
    }
    return LinearInfluenceResult(
        influence=influence,
        covariance=covariance,
        full_influence=full_influence,
        full_covariance=full_covariance,
        diagnostics=diagnostics,
    )


def coordinatewise_score_influence(
    x: np.ndarray,
    score: np.ndarray,
    information: np.ndarray,
    *,
    center_x: bool = True,
) -> LinearInfluenceResult:
    """Build a CW-ASM-style IF from externally estimated scores/information."""

    x_np = np.asarray(x, dtype=np.float64)
    psi = np.asarray(score, dtype=np.float64)
    info = np.asarray(information, dtype=np.float64).reshape(-1)
    if psi.ndim != 2 or psi.shape[0] != x_np.shape[0]:
        raise ValueError("score must have shape (n, d)")
    if len(info) != psi.shape[1] or np.any(info <= 0.0):
        raise ValueError("information must contain one positive value per response")
    design = x_np - x_np.mean(axis=0, keepdims=True) if center_x else x_np
    n, p = design.shape
    d = psi.shape[1]
    sigma_x = design.T @ design / n
    bread = np.kron(sigma_x, np.diag(info))
    equations = vec_coefficient_batch(
        np.einsum("nd,np->ndp", psi, design)
    )
    equations -= equations.mean(axis=0, keepdims=True)
    influence = np.linalg.solve(bread, equations.T).T
    covariance = influence_covariance(influence, center=False)
    diagnostics = {
        "score": "external_coordinatewise_score",
        "n": n,
        "predictor_dimension": p,
        "response_dimension": d,
        "parameter_dimension": d * p,
        "bread_condition_number": float(np.linalg.cond(bread)),
        "influence_mean_norm": float(np.linalg.norm(influence.mean(axis=0))),
        "vectorization": COEFFICIENT_VECTORIZATION,
    }
    return LinearInfluenceResult(
        influence=influence,
        covariance=covariance,
        full_influence=influence,
        full_covariance=covariance,
        diagnostics=diagnostics,
    )
