"""Formal inference adapters for coordinatewise ASM fits.

The R package ``asm`` fits one response coordinate at a time.  Its
``Cov_mat`` is the package-authoritative marginal covariance of
``sqrt(n) * (theta_hat - theta_0)``.  Cross-response covariance is not
reported by the package; this module optionally reconstructs it from the
joint observation-level score vectors while preserving those two meanings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from otrank.estimators.comparison import (
    ASMConfig,
    ASMFitInferenceResult,
    fit_coordinatewise_asm_with_inference,
)

from .covariance import influence_covariance
from otrank.vectorization import (
    COEFFICIENT_VECTORIZATION,
    predictor_indices_for_response,
    vec_coefficient_batch,
)


@dataclass(frozen=True)
class ASMInferenceResult:
    """Coordinatewise ASM package output and optional joint-score sandwich."""

    fit_state: ASMFitInferenceResult | None
    joint_influence: np.ndarray
    joint_covariance: np.ndarray
    package_marginal_block_covariance: np.ndarray
    diagnostics: dict[str, Any]


def _numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _block_diagonal(blocks: np.ndarray) -> np.ndarray:
    values = np.asarray(blocks, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("covariance blocks must have shape (d, p, p)")
    d, p, _ = values.shape
    result = np.zeros((d * p, d * p), dtype=np.float64)
    for j, block in enumerate(values):
        indices = predictor_indices_for_response(j, d, p)
        result[np.ix_(indices, indices)] = block
    return result


def _inference_arrays(
    fit_state: ASMFitInferenceResult | Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(fit_state, ASMFitInferenceResult):
        return (
            _numpy(fit_state.score_values),
            _numpy(fit_state.information_matrices),
            _numpy(fit_state.package_slope_covariance),
        )

    state = fit_state.get("asm_inference", fit_state)
    required = {
        "score_values",
        "information_matrices",
        "package_slope_covariance",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"saved ASM state is missing: {', '.join(missing)}")
    return (
        _numpy(state["score_values"]),
        _numpy(state["information_matrices"]),
        _numpy(state["package_slope_covariance"]),
    )


def asm_inference_from_fit_state(
    x: np.ndarray | torch.Tensor,
    fit_state: ASMFitInferenceResult | Mapping[str, Any],
    *,
    center_x: bool = True,
) -> ASMInferenceResult:
    """Construct the joint coordinatewise ASM influence matrix.

    The returned ``joint_covariance`` estimates cross-response blocks from
    simultaneous observation scores.  The separate
    ``package_marginal_block_covariance`` contains the exact response-wise
    ``Cov_mat`` blocks returned by ``asm`` and zeros elsewhere; its zero
    off-diagonal blocks are placeholders, not an independence assertion.
    """

    x_np = _numpy(x)
    score, information_matrices, package_blocks = _inference_arrays(fit_state)
    if x_np.ndim != 2:
        raise ValueError("x must have shape (n, p)")
    if score.ndim != 2 or score.shape[0] != x_np.shape[0]:
        raise ValueError("ASM score_values must have shape (n, d)")
    n, p = x_np.shape
    d = score.shape[1]
    if information_matrices.shape != (d, p, p):
        raise ValueError(
            "ASM information_matrices must have shape "
            f"{(d, p, p)}, got {information_matrices.shape}"
        )
    if package_blocks.shape != (d, p, p):
        raise ValueError(
            "ASM package_slope_covariance must have shape "
            f"{(d, p, p)}, got {package_blocks.shape}"
        )

    design = x_np - x_np.mean(axis=0, keepdims=True) if center_x else x_np
    influence_blocks: list[np.ndarray] = []
    block_conditions: list[float] = []
    for j in range(d):
        equations = score[:, j, None] * design
        equations -= equations.mean(axis=0, keepdims=True)
        bread = information_matrices[j]
        influence_blocks.append(np.linalg.solve(bread, equations.T).T)
        block_conditions.append(float(np.linalg.cond(bread)))

    joint_influence = vec_coefficient_batch(np.stack(influence_blocks, axis=1))
    joint_covariance = influence_covariance(joint_influence, center=False)
    package_block_covariance = _block_diagonal(package_blocks)

    relative_block_differences: list[float] = []
    for j in range(d):
        indices = predictor_indices_for_response(j, d, p)
        empirical_block = joint_covariance[np.ix_(indices, indices)]
        denominator = max(float(np.linalg.norm(package_blocks[j], ord="fro")), 1e-15)
        relative_block_differences.append(
            float(np.linalg.norm(empirical_block - package_blocks[j], ord="fro") / denominator)
        )

    diagnostics = {
        "covariance_convention": "sqrt_n_asymptotic_covariance",
        "joint_covariance_source": "empirical_cross_response_score_sandwich",
        "package_covariance_source": "R_asm_Cov_mat",
        "package_off_diagonal_blocks": "not_estimated",
        "center_x": bool(center_x),
        "n": n,
        "predictor_dimension": p,
        "response_dimension": d,
        "parameter_dimension": d * p,
        "information_condition_numbers": block_conditions,
        "relative_empirical_to_package_block_frobenius": relative_block_differences,
        "joint_influence_mean_norm": float(np.linalg.norm(joint_influence.mean(axis=0))),
        "vectorization": COEFFICIENT_VECTORIZATION,
    }
    return ASMInferenceResult(
        fit_state=fit_state if isinstance(fit_state, ASMFitInferenceResult) else None,
        joint_influence=joint_influence,
        joint_covariance=joint_covariance,
        package_marginal_block_covariance=package_block_covariance,
        diagnostics=diagnostics,
    )


def coordinatewise_asm_inference(
    x: torch.Tensor,
    y: torch.Tensor,
    config: ASMConfig | None = None,
    *,
    center_x: bool = True,
) -> ASMInferenceResult:
    """Fit coordinatewise ASM once and return point and inference outputs."""

    fit_state = fit_coordinatewise_asm_with_inference(x, y, config)
    return asm_inference_from_fit_state(x, fit_state, center_x=center_x)
