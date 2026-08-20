"""Covariance summaries shared by simulation and real-data inference.

The default convention is that an observation-level influence matrix has
shape ``(n, q)`` and estimates the influence function of
``sqrt(n) * (theta_hat - theta_0)``.  Consequently,
``influence.T @ influence / n`` is an asymptotic covariance, not the finite
sample covariance of ``theta_hat``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from otrank.vectorization import vec_coefficient_batch


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix as float64."""

    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("covariance must be a square matrix")
    return 0.5 * (value + value.T)


def influence_covariance(
    influence: np.ndarray,
    *,
    center: bool = True,
) -> np.ndarray:
    """Estimate the asymptotic covariance from observation-level influences."""

    values = np.asarray(influence, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("influence must have shape (n, parameter_dimension)")
    if not np.isfinite(values).all():
        raise ValueError("influence contains non-finite values")
    if center:
        values = values - values.mean(axis=0, keepdims=True)
    covariance = values.T @ values / values.shape[0]
    return symmetrize(covariance)


def average_influence_functions(influences: np.ndarray) -> np.ndarray:
    """Average reference-wise IFs for an ensemble estimator.

    ``influences`` must have shape ``(K, n, q)``.  Averaging IFs before
    forming the covariance retains all cross-reference covariance induced by
    the shared observations.
    """

    values = np.asarray(influences, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("influences must have shape (K, n, parameter_dimension)")
    if not np.isfinite(values).all():
        raise ValueError("influences contain non-finite values")
    return values.mean(axis=0)


def ensemble_influence_covariance(influences: np.ndarray) -> np.ndarray:
    """Return the covariance of an equally weighted reference ensemble."""

    return influence_covariance(average_influence_functions(influences))


def monte_carlo_parameter_covariance(
    estimates: np.ndarray,
    truths: np.ndarray,
    *,
    n_train: int,
) -> np.ndarray:
    """Estimate ``Cov(sqrt(n) * vec(theta_hat - theta_0))`` across reps."""

    fitted = np.asarray(estimates, dtype=np.float64)
    target = np.asarray(truths, dtype=np.float64)
    if fitted.shape != target.shape or fitted.ndim not in {2, 3}:
        raise ValueError("estimates and truths must have the same (R, ...) shape")
    if fitted.shape[0] < 2:
        raise ValueError("at least two Monte Carlo replications are required")
    raw_errors = fitted - target
    errors = (
        vec_coefficient_batch(raw_errors)
        if raw_errors.ndim == 3
        else raw_errors
    )
    centered = errors - errors.mean(axis=0, keepdims=True)
    covariance = float(n_train) * centered.T @ centered / (len(errors) - 1)
    return symmetrize(covariance)


def monte_carlo_kronecker_covariance(
    errors: np.ndarray,
    design_second_moment: np.ndarray,
    *,
    n_train: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a full parameter covariance under a Kronecker structure.

    Parameters
    ----------
    errors:
        Coefficient errors with shape ``(R, d, p)``. The returned parameter
        covariance follows standard column-major ``vec(B)`` ordering.
    design_second_moment:
        The positive-definite ``p x p`` matrix ``E[XX^T]``.

    Returns
    -------
    parameter_covariance, response_covariance
        ``parameter_covariance`` equals
        ``kron(inv(design_second_moment), response_covariance)``.  This is the
        pre-specified full-rank MC summary used when ``R - 1 < p*d``.
    """

    values = np.asarray(errors, dtype=np.float64)
    sigma_x = symmetrize(design_second_moment)
    if values.ndim != 3:
        raise ValueError("errors must have shape (R, d, p)")
    reps, _, predictors = values.shape
    if reps < 2:
        raise ValueError("at least two Monte Carlo replications are required")
    if sigma_x.shape != (predictors, predictors):
        raise ValueError("design_second_moment has the wrong shape")
    if np.linalg.eigvalsh(sigma_x)[0] <= 0.0:
        raise ValueError("design_second_moment must be positive definite")

    centered = values - values.mean(axis=0, keepdims=True)
    response_covariance = float(n_train) * np.einsum(
        "rdp,pq,req->de",
        centered,
        sigma_x,
        centered,
        optimize=True,
    ) / (predictors * (reps - 1))
    response_covariance = symmetrize(response_covariance)
    parameter_covariance = np.kron(
        np.linalg.inv(sigma_x),
        response_covariance,
    )
    return symmetrize(parameter_covariance), response_covariance


def case_weight_plugin_score_decomposition(
    plugin_scores: np.ndarray,
    case_weight_scores: np.ndarray,
    hessian: np.ndarray,
    *,
    ridge: float = 0.0,
    top_eigenvectors: int = 10,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Decompose case-weight scores into plug-in score plus OT feedback.

    ``plugin_scores`` is ``P`` and ``case_weight_scores`` is ``D=-C``. Both are
    centered internally.  All meat contributions are propagated through the
    same supplied Hessian inverse, isolating score feedback from curvature.
    """

    plugin = np.asarray(plugin_scores, dtype=np.float64)
    case_weight = np.asarray(case_weight_scores, dtype=np.float64)
    bread = symmetrize(hessian)
    if plugin.shape != case_weight.shape or plugin.ndim != 2:
        raise ValueError(
            "plugin_scores and case_weight_scores must have equal (n, q) shape"
        )
    if bread.shape != (plugin.shape[1], plugin.shape[1]):
        raise ValueError("hessian dimension does not match score width")
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")
    if ridge:
        bread = bread + float(ridge) * np.eye(len(bread))
    plugin = plugin - plugin.mean(axis=0, keepdims=True)
    case_weight = case_weight - case_weight.mean(axis=0, keepdims=True)
    feedback = case_weight - plugin
    n, q = plugin.shape

    plugin_norm_sq = float(np.sum(plugin * plugin))
    case_weight_norm_sq = float(np.sum(case_weight * case_weight))
    kappa = (
        float(np.sum(case_weight * plugin) / plugin_norm_sq)
        if plugin_norm_sq > 0.0
        else math.nan
    )
    scalar_residual_sq = (
        float(
            np.sum((case_weight - kappa * plugin) ** 2) / case_weight_norm_sq
        )
        if case_weight_norm_sq > 0.0 and math.isfinite(kappa)
        else math.nan
    )

    meat_plugin = plugin.T @ plugin / float(n)
    meat_cross = (plugin.T @ feedback + feedback.T @ plugin) / float(n)
    meat_feedback = feedback.T @ feedback / float(n)
    meat_case_weight = case_weight.T @ case_weight / float(n)
    reconstructed_meat = meat_plugin + meat_cross + meat_feedback

    inverse = np.linalg.inv(bread)

    def propagate(meat: np.ndarray) -> np.ndarray:
        return symmetrize(inverse @ meat @ inverse.T)

    covariance_plugin = propagate(meat_plugin)
    covariance_cross = propagate(meat_cross)
    covariance_feedback = propagate(meat_feedback)
    covariance_case_weight = propagate(meat_case_weight)
    covariance_reconstructed = (
        covariance_plugin + covariance_cross + covariance_feedback
    )

    plugin_scale = np.sqrt(np.sum(plugin * plugin, axis=0))
    case_weight_scale = np.sqrt(np.sum(case_weight * case_weight, axis=0))
    coordinate_correlation = np.divide(
        np.sum(plugin * case_weight, axis=0),
        plugin_scale * case_weight_scale,
        out=np.full(q, np.nan),
        where=(plugin_scale > 0.0) & (case_weight_scale > 0.0),
    )
    plugin_se = np.sqrt(np.clip(np.diag(covariance_plugin), 0.0, None))
    case_weight_se = np.sqrt(
        np.clip(np.diag(covariance_case_weight), 0.0, None)
    )
    se_ratio = np.divide(
        case_weight_se,
        plugin_se,
        out=np.full(q, np.nan),
        where=plugin_se > 0.0,
    )

    plugin_eigenvalues, plugin_eigenvectors = np.linalg.eigh(covariance_plugin)
    case_weight_eigenvalues, case_weight_eigenvectors = np.linalg.eigh(
        covariance_case_weight
    )
    plugin_order = np.argsort(plugin_eigenvalues)[::-1]
    case_weight_order = np.argsort(case_weight_eigenvalues)[::-1]
    plugin_eigenvalues = plugin_eigenvalues[plugin_order]
    case_weight_eigenvalues = case_weight_eigenvalues[case_weight_order]
    plugin_eigenvectors = plugin_eigenvectors[:, plugin_order]
    case_weight_eigenvectors = case_weight_eigenvectors[:, case_weight_order]
    top = max(1, min(int(top_eigenvectors), q))
    singular_values = np.linalg.svd(
        plugin_eigenvectors[:, :top].T @ case_weight_eigenvectors[:, :top],
        compute_uv=False,
    )
    principal_angles_deg = np.degrees(
        np.arccos(np.clip(singular_values, -1.0, 1.0))
    )

    hessian_eigenvalues, hessian_eigenvectors = np.linalg.eigh(bread)
    plugin_variance_hessian_basis = np.diag(
        hessian_eigenvectors.T @ covariance_plugin @ hessian_eigenvectors
    )
    case_weight_variance_hessian_basis = np.diag(
        hessian_eigenvectors.T @ covariance_case_weight @ hessian_eigenvectors
    )
    hessian_direction_se_ratio = np.sqrt(
        np.divide(
            case_weight_variance_hessian_basis,
            plugin_variance_hessian_basis,
            out=np.full(q, np.nan),
            where=plugin_variance_hessian_basis > 0.0,
        )
    )
    valid_direction = (
        np.isfinite(hessian_direction_se_ratio)
        & (hessian_direction_se_ratio > 0.0)
        & (np.abs(hessian_eigenvalues) > 0.0)
    )
    curvature_log_se_ratio_correlation = (
        float(
            np.corrcoef(
                np.log(np.abs(hessian_eigenvalues[valid_direction])),
                np.log(hessian_direction_se_ratio[valid_direction]),
            )[0, 1]
        )
        if np.sum(valid_direction) >= 2
        else math.nan
    )

    finite_coordinate_correlation = coordinate_correlation[
        np.isfinite(coordinate_correlation)
    ]
    finite_se_ratio = se_ratio[np.isfinite(se_ratio)]
    trace_plugin = float(np.trace(covariance_plugin) / q)
    trace_cross = float(np.trace(covariance_cross) / q)
    trace_feedback = float(np.trace(covariance_feedback) / q)
    trace_case_weight = float(np.trace(covariance_case_weight) / q)
    summary = {
        "case_count": int(n),
        "parameter_dimension": int(q),
        "scalar_projection_kappa": kappa,
        "scalar_projection_relative_residual_sq": scalar_residual_sq,
        "coordinate_correlation_median": (
            float(np.median(finite_coordinate_correlation))
            if len(finite_coordinate_correlation)
            else math.nan
        ),
        "coordinate_correlation_q25": (
            float(np.quantile(finite_coordinate_correlation, 0.25))
            if len(finite_coordinate_correlation)
            else math.nan
        ),
        "coordinate_correlation_q75": (
            float(np.quantile(finite_coordinate_correlation, 0.75))
            if len(finite_coordinate_correlation)
            else math.nan
        ),
        "trace_plugin_per_parameter": trace_plugin,
        "trace_cross_per_parameter": trace_cross,
        "trace_feedback_per_parameter": trace_feedback,
        "trace_case_weight_per_parameter": trace_case_weight,
        "trace_cross_fraction_of_case_weight": (
            trace_cross / trace_case_weight
            if trace_case_weight != 0.0
            else math.nan
        ),
        "trace_feedback_fraction_of_case_weight": (
            trace_feedback / trace_case_weight
            if trace_case_weight != 0.0
            else math.nan
        ),
        "case_weight_to_plugin_trace_ratio": (
            trace_case_weight / trace_plugin if trace_plugin != 0.0 else math.nan
        ),
        "case_weight_to_plugin_se_ratio_median": (
            float(np.median(finite_se_ratio)) if len(finite_se_ratio) else math.nan
        ),
        "case_weight_to_plugin_se_ratio_q25": (
            float(np.quantile(finite_se_ratio, 0.25)) if len(finite_se_ratio) else math.nan
        ),
        "case_weight_to_plugin_se_ratio_q75": (
            float(np.quantile(finite_se_ratio, 0.75)) if len(finite_se_ratio) else math.nan
        ),
        "top_eigenspace_dimension": int(top),
        "top_eigenspace_principal_angle_mean_deg": float(np.mean(principal_angles_deg)),
        "top_eigenspace_principal_angle_max_deg": float(np.max(principal_angles_deg)),
        "relative_covariance_frobenius_difference": float(
            np.linalg.norm(covariance_case_weight - covariance_plugin)
            / max(np.linalg.norm(covariance_plugin), np.finfo(float).tiny)
        ),
        "meat_reconstruction_relative_error": float(
            np.linalg.norm(meat_case_weight - reconstructed_meat)
            / max(np.linalg.norm(meat_case_weight), np.finfo(float).tiny)
        ),
        "covariance_reconstruction_relative_error": float(
            np.linalg.norm(covariance_case_weight - covariance_reconstructed)
            / max(np.linalg.norm(covariance_case_weight), np.finfo(float).tiny)
        ),
        "hessian_abs_minimum_eigenvalue": float(np.min(np.abs(hessian_eigenvalues))),
        "hessian_abs_maximum_eigenvalue": float(np.max(np.abs(hessian_eigenvalues))),
        "hessian_condition_number": float(np.linalg.cond(bread)),
        "log_curvature_vs_log_se_ratio_correlation": curvature_log_se_ratio_correlation,
    }
    arrays = {
        "plugin_scores": plugin,
        "case_weight_scores": case_weight,
        "feedback_scores": feedback,
        "meat_plugin": meat_plugin,
        "meat_cross": meat_cross,
        "meat_feedback": meat_feedback,
        "meat_case_weight": meat_case_weight,
        "covariance_plugin": covariance_plugin,
        "covariance_cross": covariance_cross,
        "covariance_feedback": covariance_feedback,
        "covariance_case_weight": covariance_case_weight,
        "coordinate_correlation": coordinate_correlation,
        "case_weight_to_plugin_se_ratio": se_ratio,
        "plugin_covariance_eigenvalues_desc": plugin_eigenvalues,
        "case_weight_covariance_eigenvalues_desc": case_weight_eigenvalues,
        "top_eigenspace_principal_angles_deg": principal_angles_deg,
        "hessian_eigenvalues": hessian_eigenvalues,
        "plugin_variance_in_hessian_basis": plugin_variance_hessian_basis,
        "case_weight_variance_in_hessian_basis": case_weight_variance_hessian_basis,
        "hessian_direction_se_ratio": hessian_direction_se_ratio,
    }
    return summary, arrays
