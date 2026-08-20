"""Wald summaries for asymptotic covariance matrices."""

from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np

from .covariance import symmetrize
from otrank.vectorization import unvec_coefficient, vec_coefficient


def _vectorize_estimate(value: np.ndarray) -> np.ndarray:
    if value.ndim == 1:
        return value
    if value.ndim == 2:
        return np.asarray(vec_coefficient(value), dtype=np.float64)
    raise ValueError("estimate must be a parameter vector or a (d, p) matrix")


def _critical_value(alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


def marginal_standard_errors(
    asymptotic_covariance: np.ndarray,
    *,
    n_train: int,
) -> np.ndarray:
    """Return coefficient SEs when ``V`` is covariance of ``sqrt(n)`` error."""

    if n_train < 1:
        raise ValueError("n_train must be positive")
    covariance = symmetrize(asymptotic_covariance)
    diagonal = np.diag(covariance)
    tolerance = max(float(np.max(np.abs(diagonal))), 1.0) * 1e-12
    if np.min(diagonal) < -tolerance:
        raise ValueError("asymptotic covariance has a negative diagonal entry")
    return np.sqrt(np.maximum(diagonal, 0.0) / float(n_train))


def coefficient_wald_intervals(
    estimate: np.ndarray,
    asymptotic_covariance: np.ndarray,
    *,
    n_train: int,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Return elementwise Wald lower and upper endpoints."""

    values = np.asarray(estimate, dtype=np.float64)
    standard_errors = marginal_standard_errors(
        asymptotic_covariance,
        n_train=n_train,
    )
    if values.size != len(standard_errors):
        raise ValueError("estimate size must match covariance dimension")
    half_width = _critical_value(alpha) * standard_errors
    flat = _vectorize_estimate(values)
    lower = flat - half_width
    upper = flat + half_width
    if values.ndim == 2:
        d, p = values.shape
        return (
            np.asarray(unvec_coefficient(lower, d, p)),
            np.asarray(unvec_coefficient(upper, d, p)),
        )
    return lower, upper


def contrast_wald_interval(
    estimate: np.ndarray,
    asymptotic_covariance: np.ndarray,
    contrast: np.ndarray,
    *,
    n_train: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Return a Wald interval for ``a.T @ vec(estimate)``."""

    raw_values = np.asarray(estimate, dtype=np.float64)
    values = _vectorize_estimate(raw_values)
    direction = _vectorize_estimate(np.asarray(contrast, dtype=np.float64))
    covariance = symmetrize(asymptotic_covariance)
    if direction.shape != values.shape or covariance.shape != (
        len(values),
        len(values),
    ):
        raise ValueError("estimate, contrast, and covariance dimensions must agree")
    variance = float(direction @ covariance @ direction / float(n_train))
    if variance < -1e-12:
        raise ValueError("contrast variance is negative")
    standard_error = float(np.sqrt(max(variance, 0.0)))
    center = float(direction @ values)
    half_width = _critical_value(alpha) * standard_error
    return {
        "estimate": center,
        "standard_error": standard_error,
        "lower": center - half_width,
        "upper": center + half_width,
        "alpha": float(alpha),
    }


def marginal_wald_summary(
    estimate: np.ndarray,
    truth: np.ndarray,
    asymptotic_covariance: np.ndarray,
    *,
    n_train: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Return coefficientwise coverage and interval-width summaries."""

    fitted = np.asarray(estimate, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if fitted.shape != target.shape:
        raise ValueError("estimate and truth must have the same shape")
    lower, upper = coefficient_wald_intervals(
        fitted,
        asymptotic_covariance,
        n_train=n_train,
        alpha=alpha,
    )
    covered = (target >= lower) & (target <= upper)
    half_width = 0.5 * (upper - lower)
    return {
        "alpha": float(alpha),
        "nominal_coverage": float(1.0 - alpha),
        "marginal_coverage": float(np.mean(covered)),
        "average_half_width": float(np.mean(half_width)),
        "average_interval_length": float(2.0 * np.mean(half_width)),
        "covered": covered,
        "lower": lower,
        "upper": upper,
    }
