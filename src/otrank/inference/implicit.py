"""Implicit differentiation for case-weighted estimating equations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from otrank.vectorization import (
    COEFFICIENT_VECTORIZATION,
    unvec_coefficient,
    vec_coefficient,
)


def influence_from_scores_and_hessian(
    hessian: np.ndarray,
    case_scores: np.ndarray,
    *,
    ridge: float = 0.0,
    center: bool = True,
) -> np.ndarray:
    """Return ``H^{-1} psi_i`` influences from positive-sign case scores.

    ``case_scores`` has shape ``(n, q)`` and follows the OT-rank paper's
    convention that the empirical parameter gradient is ``-mean(psi_i)``.
    Thus the influence has a positive sign after solving by the Hessian.
    """

    bread = np.asarray(hessian, dtype=np.float64)
    scores = np.asarray(case_scores, dtype=np.float64)
    if bread.ndim != 2 or bread.shape[0] != bread.shape[1]:
        raise ValueError("hessian must be square")
    if scores.ndim != 2 or scores.shape[1] != bread.shape[0]:
        raise ValueError("case_scores must have shape (n, parameter_dimension)")
    if scores.shape[0] < 1 or not np.isfinite(scores).all():
        raise ValueError("case_scores must be nonempty and finite")
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")
    bread = 0.5 * (bread + bread.T)
    if ridge:
        bread = bread + float(ridge) * np.eye(len(bread))
    if center:
        scores = scores - scores.mean(axis=0, keepdims=True)
    influence = np.linalg.solve(bread, scores.T).T
    if center:
        influence -= influence.mean(axis=0, keepdims=True)
    return influence


def implicit_influence_from_derivatives(
    hessian: np.ndarray,
    case_score_jacobian: np.ndarray,
    case_weights: np.ndarray,
    *,
    ridge: float = 0.0,
) -> np.ndarray:
    """Solve ``IF_i = -A^{-1} C_i`` for contamination-weight derivatives.

    The Jacobian has shape ``(q, n)`` and differentiates the parameter score
    with respect to raw case weights.  Column centering by the empirical case
    weights converts those derivatives to contamination paths
    ``delta_i - F_n``.
    """

    bread = np.asarray(hessian, dtype=np.float64)
    jacobian = np.asarray(case_score_jacobian, dtype=np.float64)
    weights = np.asarray(case_weights, dtype=np.float64).reshape(-1)
    if bread.ndim != 2 or bread.shape[0] != bread.shape[1]:
        raise ValueError("hessian must be square")
    if jacobian.shape != (bread.shape[0], len(weights)):
        raise ValueError("case_score_jacobian must have shape (q, n)")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("case_weights must sum to one")
    if np.any(weights < 0.0):
        raise ValueError("case_weights must be nonnegative")
    bread = 0.5 * (bread + bread.T)
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")
    if ridge:
        bread = bread + float(ridge) * np.eye(len(bread))
    contamination_derivative = jacobian - (jacobian @ weights)[:, None]
    influence = -np.linalg.solve(bread, contamination_derivative).T
    influence -= influence.mean(axis=0, keepdims=True)
    return influence


def implicit_influence_from_contamination_derivatives(
    hessian: np.ndarray,
    contamination_derivative: np.ndarray,
    *,
    ridge: float = 0.0,
) -> np.ndarray:
    """Solve ``IF_i = -A^{-1} C_i`` when ``C`` is already centered."""

    bread = np.asarray(hessian, dtype=np.float64)
    derivative = np.asarray(contamination_derivative, dtype=np.float64)
    if bread.ndim != 2 or bread.shape[0] != bread.shape[1]:
        raise ValueError("hessian must be square")
    if derivative.ndim != 2 or derivative.shape[0] != bread.shape[0]:
        raise ValueError("contamination_derivative must have shape (q, n)")
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")
    bread = 0.5 * (bread + bread.T)
    if ridge:
        bread = bread + float(ridge) * np.eye(len(bread))
    influence = -np.linalg.solve(bread, derivative).T
    influence -= influence.mean(axis=0, keepdims=True)
    return influence


def autograd_case_weight_influence(
    objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    parameter: torch.Tensor,
    case_weights: torch.Tensor,
    *,
    ridge: float = 0.0,
    vectorize_derivatives: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Differentiate a scalar weighted objective with respect to case weights.

    ``objective(parameter, raw_case_weights)`` should normalize the raw weights
    internally.  The returned IF has shape ``(n, parameter.numel())``.
    """

    hessian_np, contamination_derivative, diagnostics = (
        autograd_case_weight_score_derivatives(
            objective,
            parameter,
            case_weights,
            vectorize_derivatives=vectorize_derivatives,
        )
    )
    influence = implicit_influence_from_contamination_derivatives(
        hessian_np,
        contamination_derivative,
        ridge=ridge,
    )
    diagnostics.update(
        {
            "influence_mean_norm": float(np.linalg.norm(influence.mean(axis=0))),
            "local_root_interpretation": True,
        }
    )
    return influence, diagnostics


def autograd_case_weight_score_derivatives(
    objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    parameter: torch.Tensor,
    case_weights: torch.Tensor,
    *,
    vectorize_derivatives: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the empirical Hessian and contamination score derivatives.

    The second returned array is ``C`` in ``IF_i = -A^{-1} C_i`` and has
    shape ``(q, n)``.  It is exposed separately so diagnostics can compare
    the case-weight score ``D_i = -C_i`` with the plug-in score without
    recomputing the higher-order derivatives.
    """

    parameter_shape = tuple(int(size) for size in parameter.shape)
    if parameter.ndim == 2:
        response_dimension, predictor_dimension = parameter_shape
        theta = vec_coefficient(parameter.detach()).clone().requires_grad_(True)

        def restore_parameter(vector: torch.Tensor) -> torch.Tensor:
            return unvec_coefficient(
                vector,
                response_dimension,
                predictor_dimension,
            )

        vectorization = COEFFICIENT_VECTORIZATION
    else:
        theta = parameter.detach().clone().reshape(-1).requires_grad_(True)

        def restore_parameter(vector: torch.Tensor) -> torch.Tensor:
            return vector.reshape(parameter_shape)

        vectorization = "native torch flatten (non-matrix parameter)"

    weights = case_weights.detach().clone().to(
        device=theta.device, dtype=theta.dtype
    ).requires_grad_(True)
    if weights.ndim != 1 or len(weights) < 1:
        raise ValueError("case_weights must be a nonempty vector")
    normalized = weights.detach().cpu().double().numpy()
    normalized /= normalized.sum()

    hessian = torch.autograd.functional.hessian(
        lambda current: objective(restore_parameter(current), weights),
        theta,
        vectorize=vectorize_derivatives,
    )

    theta_for_weights = theta.detach().clone().requires_grad_(True)

    def score_by_weight(current_weights: torch.Tensor) -> torch.Tensor:
        loss = objective(restore_parameter(theta_for_weights), current_weights)
        score = torch.autograd.grad(
            loss,
            theta_for_weights,
            create_graph=True,
        )[0]
        return score

    jacobian = torch.autograd.functional.jacobian(
        score_by_weight,
        weights,
        vectorize=vectorize_derivatives,
    )
    hessian_np = hessian.detach().cpu().double().numpy()
    jacobian_np = jacobian.detach().cpu().double().numpy()
    contamination_derivative = jacobian_np - (jacobian_np @ normalized)[:, None]
    symmetric_hessian = 0.5 * (hessian_np + hessian_np.T)
    eigenvalues = np.linalg.eigvalsh(symmetric_hessian)
    score = torch.autograd.grad(
        objective(restore_parameter(theta), weights),
        theta,
    )[0]
    diagnostics = {
        "parameter_dimension": theta.numel(),
        "case_count": len(weights),
        "score_norm": float(torch.linalg.vector_norm(score).detach().cpu()),
        "hessian_minimum_eigenvalue": float(eigenvalues[0]),
        "hessian_maximum_eigenvalue": float(eigenvalues[-1]),
        "hessian_condition_number": float(np.linalg.cond(symmetric_hessian)),
        "vectorized_derivatives": bool(vectorize_derivatives),
        "vectorization": vectorization,
    }
    return hessian_np, contamination_derivative, diagnostics
