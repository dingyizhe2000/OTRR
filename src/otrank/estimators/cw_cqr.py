"""Coordinatewise composite-quantile regression inference utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, norm

from otrank.vectorization import (
    COEFFICIENT_VECTORIZATION,
    response_indices_for_predictor,
    unvec_coefficient,
    vec_coefficient,
    vec_coefficient_batch,
)


@dataclass(frozen=True)
class CWCQRInference:
    covariance: np.ndarray
    influence: np.ndarray
    density_at_quantiles: np.ndarray
    bread_condition_numbers: np.ndarray
    bread_ranks: np.ndarray
    vectorization: str = COEFFICIENT_VECTORIZATION


def _gaussian_density_at_points(values: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    points = np.asarray(points, dtype=np.float64).reshape(-1)
    n = values.size
    if n < 3:
        raise ValueError("At least three residuals are required")
    std = float(np.std(values, ddof=1))
    iqr_scale = float((np.quantile(values, 0.75) - np.quantile(values, 0.25)) / 1.349)
    candidates = [scale for scale in (std, iqr_scale) if np.isfinite(scale) and scale > 0]
    scale = min(candidates) if candidates else max(float(np.max(np.abs(values))), 1.0)
    bandwidth = max(0.9 * scale * n ** (-0.2), np.finfo(np.float64).eps ** 0.25)
    standardized = (values[:, None] - points[None, :]) / bandwidth
    return np.mean(np.exp(-0.5 * standardized**2), axis=0) / (
        bandwidth * np.sqrt(2.0 * np.pi)
    )


def coordinate_cqr_influence(
    x: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    intercepts: np.ndarray,
    tau: np.ndarray,
    *,
    derivative_ridge: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Return slope IFs for one response, density estimates, condition, and rank.

    The nuisance quantile intercepts are included in the estimating equation and
    profiled through the full ``(L + p)`` bread before the slope rows are kept.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    beta = np.asarray(beta, dtype=np.float64).reshape(-1)
    intercepts = np.asarray(intercepts, dtype=np.float64).reshape(-1)
    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    n, p = x.shape
    levels = tau.size
    if y.shape != (n,) or beta.shape != (p,) or intercepts.shape != (levels,):
        raise ValueError("Incompatible CQR input shapes")

    base_residual = y - x @ beta
    expanded_residual = base_residual[:, None] - intercepts[None, :]
    psi = tau[None, :] - (expanded_residual < 0.0).astype(np.float64)
    score = np.concatenate((psi, x * psi.sum(axis=1, keepdims=True)), axis=1)
    score -= score.mean(axis=0, keepdims=True)

    density = _gaussian_density_at_points(base_residual, intercepts)
    mean_x = x.mean(axis=0)
    second_x = x.T @ x / n
    bread = np.zeros((levels + p, levels + p), dtype=np.float64)
    bread[:levels, :levels] = np.diag(density)
    bread[:levels, levels:] = density[:, None] * mean_x[None, :]
    bread[levels:, :levels] = bread[:levels, levels:].T
    bread[levels:, levels:] = density.sum() * second_x

    ridge = float(derivative_ridge) * max(float(np.trace(bread)) / bread.shape[0], 1.0)
    bread_regularized = bread + ridge * np.eye(bread.shape[0])
    condition = float(np.linalg.cond(bread_regularized))
    rank = int(np.linalg.matrix_rank(bread_regularized))
    influence = np.linalg.solve(bread_regularized, score.T).T[:, levels:]
    return influence, density, condition, rank


def cw_cqr_covariance(
    x: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    intercepts: np.ndarray,
    tau: np.ndarray,
    *,
    derivative_ridge: float = 1e-10,
) -> CWCQRInference:
    """Estimate covariance for standard column-major ``sqrt(n) vec(Bhat)``."""

    y = np.asarray(y, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    intercepts = np.asarray(intercepts, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("y must have shape (n, d)")
    n, d = y.shape
    p = np.asarray(x).shape[1]
    if beta.shape != (d, p):
        raise ValueError("beta must have shape (d, p)")
    if intercepts.shape != (d, np.asarray(tau).size):
        raise ValueError("intercepts must have shape (d, L)")

    if_blocks: list[np.ndarray] = []
    densities: list[np.ndarray] = []
    conditions: list[float] = []
    ranks: list[int] = []
    for response in range(d):
        influence, density, condition, rank = coordinate_cqr_influence(
            x,
            y[:, response],
            beta[response],
            intercepts[response],
            tau,
            derivative_ridge=derivative_ridge,
        )
        if_blocks.append(influence)
        densities.append(density)
        conditions.append(condition)
        ranks.append(rank)
    joint_if = vec_coefficient_batch(np.stack(if_blocks, axis=1))
    covariance = joint_if.T @ joint_if / n
    return CWCQRInference(
        covariance=covariance,
        influence=joint_if,
        density_at_quantiles=np.stack(densities),
        bread_condition_numbers=np.asarray(conditions),
        bread_ranks=np.asarray(ranks),
    )


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    flat = values.reshape(-1)
    order = np.argsort(flat)
    ranked = flat[order]
    adjusted = ranked * flat.size / np.arange(1, flat.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(flat)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out.reshape(values.shape)


def cqr_significance_summary(
    beta: np.ndarray,
    covariance: np.ndarray,
    n: int,
) -> dict[str, object]:
    """Coefficient and predictor-level Wald summaries under standard vec(B)."""

    beta = np.asarray(beta, dtype=np.float64)
    d, p = beta.shape
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (d * p, d * p):
        raise ValueError("covariance has the wrong shape")
    flat = np.asarray(vec_coefficient(beta), dtype=np.float64)
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf) / n)
    z = np.divide(flat, se, out=np.zeros_like(flat), where=se > 0)
    coefficient_p = 2.0 * norm.sf(np.abs(z))
    coefficient_q = benjamini_hochberg(coefficient_p)

    predictor_p = np.empty(p, dtype=np.float64)
    for predictor in range(p):
        indices = response_indices_for_predictor(predictor, d, p)
        effect = flat[indices]
        estimator_cov = covariance[np.ix_(indices, indices)] / n
        rank = int(np.linalg.matrix_rank(estimator_cov))
        statistic = float(effect @ np.linalg.pinv(estimator_cov) @ effect)
        predictor_p[predictor] = float(chi2.sf(statistic, max(rank, 1)))
    predictor_q = benjamini_hochberg(predictor_p)
    return {
        "standard_errors": np.asarray(unvec_coefficient(se, d, p)),
        "coefficient_p": np.asarray(unvec_coefficient(coefficient_p, d, p)),
        "coefficient_q": np.asarray(unvec_coefficient(coefficient_q, d, p)),
        "predictor_p": predictor_p,
        "predictor_q": predictor_q,
        "coef_significant_90": int(np.sum(coefficient_p < 0.10)),
        "coef_significant_95": int(np.sum(coefficient_p < 0.05)),
        "coef_bh_q10": int(np.sum(coefficient_q < 0.10)),
        "coef_bh_q05": int(np.sum(coefficient_q < 0.05)),
        "predictor_bh_q10": int(np.sum(predictor_q < 0.10)),
        "predictor_bh_q05": int(np.sum(predictor_q < 0.05)),
    }
