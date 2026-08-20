"""Plug-in and case-weight influence functions for dense PNW fits.

The plug-in covariance is the default inference method.  The optional
case-weight local-root diagnostic uses dense PyTorch higher-order derivatives
because the production Triton backward does not provide a double backward.
Both methods condition on the finite reference clouds and tuning parameters.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch

from .covariance import ensemble_influence_covariance
from .implicit import (
    autograd_case_weight_influence,
    autograd_case_weight_score_derivatives,
    influence_from_scores_and_hessian,
)
from otrank.vectorization import COEFFICIENT_VECTORIZATION, vec_coefficient_batch
from ..ot_maps.pnw import sinkhorn_log_scalings, sinkhorn_log_scalings_triton


def _weighted_dense_pnw_state(
    beta: torch.Tensor,
    raw_case_weights: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    sinkhorn_iters: int,
    cost_scale: float,
    intercept_mode: str,
    x_mean: torch.Tensor | None = None,
    y_mean: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized weights, residuals, ranks, and score design."""

    weights = raw_case_weights / raw_case_weights.sum()
    mode = intercept_mode.lower()
    if mode == "zero":
        design = x
        residual = y - x @ beta.T
    elif mode in {"mean", "center", "intercept"}:
        weighted_x_mean = torch.sum(weights[:, None] * x, dim=0)
        weighted_y_mean = torch.sum(weights[:, None] * y, dim=0)
        design = x - weighted_x_mean[None, :]
        residual = y - weighted_y_mean[None, :] - design @ beta.T
    elif mode == "pilot":
        if x_mean is None or y_mean is None:
            raise ValueError("pilot intercept mode requires fixed x_mean and y_mean")
        fixed_x_mean = x_mean.to(device=x.device, dtype=x.dtype).reshape(1, -1)
        fixed_y_mean = y_mean.to(device=y.device, dtype=y.dtype).reshape(1, -1)
        if (
            fixed_x_mean.shape[1] != x.shape[1]
            or fixed_y_mean.shape[1] != y.shape[1]
        ):
            raise ValueError("pilot means do not match x and y dimensions")
        design = x - fixed_x_mean
        residual = y - fixed_y_mean - design @ beta.T
    else:
        raise ValueError("intercept_mode must be 'zero', 'mean', or 'pilot'")

    squared_distance = (
        torch.sum(residual.square(), dim=1, keepdim=True)
        + torch.sum(target.square(), dim=1)[None, :]
        - 2.0 * residual @ target.T
    )
    cost = 0.5 * squared_distance / float(cost_scale)
    log_kernel = -cost / float(epsilon)
    tiny = torch.finfo(weights.dtype).tiny
    log_a = torch.log(weights.clamp_min(tiny))
    log_b = torch.full(
        (target.shape[0],),
        -math.log(float(target.shape[0])),
        dtype=target.dtype,
        device=target.device,
    )
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(max(1, int(sinkhorn_iters))):
        u = log_a - torch.logsumexp(log_kernel + v[None, :], dim=1)
        v = log_b - torch.logsumexp(log_kernel + u[:, None], dim=0)
    conditional = torch.softmax(v[None, :] + log_kernel, dim=1)
    ranks = conditional @ target
    return weights, residual, ranks, design


def weighted_dense_pnw_objective(
    beta: torch.Tensor,
    raw_case_weights: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    sinkhorn_iters: int,
    cost_scale: float,
    l2_penalty: float = 0.0,
    intercept_mode: str = "zero",
    x_mean: torch.Tensor | None = None,
    y_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate the candidate-dependent PNW objective under case weights."""

    if epsilon <= 0.0 or cost_scale <= 0.0:
        raise ValueError("epsilon and cost_scale must be positive")
    if l2_penalty < 0.0:
        raise ValueError("l2_penalty must be nonnegative")
    # Expanding squared distances in the shared state helper avoids
    # ``torch.cdist``, whose supported version has no double backward.
    weights, residual, ranks, _ = _weighted_dense_pnw_state(
        beta,
        raw_case_weights,
        x,
        y,
        target,
        epsilon=epsilon,
        sinkhorn_iters=sinkhorn_iters,
        cost_scale=cost_scale,
        intercept_mode=intercept_mode,
        x_mean=x_mean,
        y_mean=y_mean,
    )
    loss = torch.sum(weights * torch.sum(residual * ranks, dim=1))
    return loss + 0.5 * float(l2_penalty) * torch.sum(beta.square())


def _uniform_design_and_residual(
    beta: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    intercept_mode: str,
    x_mean: torch.Tensor | None,
    y_mean: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mode = intercept_mode.lower()
    if mode == "zero":
        design = x
        residual = y - x @ beta.T
    elif mode in {"mean", "center", "intercept"}:
        design = x - x.mean(dim=0, keepdim=True)
        residual = y - y.mean(dim=0, keepdim=True) - design @ beta.T
    elif mode == "pilot":
        if x_mean is None or y_mean is None:
            raise ValueError("pilot intercept mode requires fixed x_mean and y_mean")
        fixed_x_mean = x_mean.to(device=x.device, dtype=x.dtype).reshape(1, -1)
        fixed_y_mean = y_mean.to(device=y.device, dtype=y.dtype).reshape(1, -1)
        design = x - fixed_x_mean
        residual = y - fixed_y_mean - design @ beta.T
    else:
        raise ValueError("intercept_mode must be 'zero', 'mean', or 'pilot'")
    return design, residual


@torch.no_grad()
def reconstruct_pnw_dual(
    residual: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    cost_scale: float,
    minimum_iterations: int = 1,
    tolerance: float | None = 1e-5,
    max_iterations: int = 200,
    check_every: int = 5,
    backend: str = "auto",
    chunk_size: int | None = 8192,
    triton_block_m: int = 1024,
    init_u: torch.Tensor | None = None,
    init_v: torch.Tensor | None = None,
    require_convergence: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Reconstruct the entropic-OT dual at a final fitted coefficient.

    Training may use one warm-started Sinkhorn update per optimizer step.  That
    update count is not an inference definition.  This routine instead solves
    the dual fixed point at the final residual cloud and records an explicit
    row-marginal convergence error.  Existing fits need no saved transient
    dual; a supplied dual is only an initializer and is refined by the same
    convergence rule.
    """

    if residual.ndim != 2 or target.ndim != 2 or residual.shape[1] != target.shape[1]:
        raise ValueError("residual and target must be two-dimensional with equal width")
    if epsilon <= 0.0 or cost_scale <= 0.0:
        raise ValueError("epsilon and cost_scale must be positive")
    minimum = max(1, int(minimum_iterations))
    maximum = int(max_iterations)
    check = max(1, int(check_every))
    if maximum < minimum:
        raise ValueError("max_iterations must be at least minimum_iterations")
    if tolerance is not None and float(tolerance) <= 0.0:
        raise ValueError("tolerance must be positive or None")

    chosen_backend = str(backend).lower()
    if chosen_backend == "auto":
        chosen_backend = (
            "triton"
            if residual.device.type == "cuda"
            and residual.dtype == torch.float32
            and target.dtype == torch.float32
            else "pytorch"
        )
    if chosen_backend not in {"pytorch", "triton"}:
        raise ValueError("backend must be 'auto', 'pytorch', or 'triton'")
    if chosen_backend == "triton" and (
        residual.device.type != "cuda"
        or residual.dtype != torch.float32
        or target.dtype != torch.float32
    ):
        raise ValueError("Triton dual reconstruction requires CUDA float32 tensors")

    chunk = target.shape[0] if chunk_size is None else max(1, int(chunk_size))
    log_a = torch.full(
        (residual.shape[0],),
        -math.log(float(residual.shape[0])),
        device=residual.device,
        dtype=residual.dtype,
    )

    def solve_batch(
        iterations: int,
        current_u: torch.Tensor | None,
        current_v: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if chosen_backend == "triton":
            return sinkhorn_log_scalings_triton(
                residual,
                target,
                epsilon=float(epsilon),
                sinkhorn_iters=int(iterations),
                cost_scale=float(cost_scale),
                block_m=int(triton_block_m),
                init_u=current_u,
                init_v=current_v,
            )
        u, v, _ = sinkhorn_log_scalings(
            residual,
            target,
            epsilon=float(epsilon),
            sinkhorn_iters=int(iterations),
            cost_scale=float(cost_scale),
            chunk_size=chunk,
            init_u=current_u,
            init_v=current_v,
        )
        return u, v

    def row_log_marginal_error(u: torch.Tensor, v: torch.Tensor) -> float:
        if chosen_backend == "triton":
            from ..ot_maps.pnw_triton import triton_row_logsumexp

            row_lse = triton_row_logsumexp(
                residual,
                target,
                v,
                epsilon=float(epsilon),
                cost_scale=float(cost_scale),
                block_m=int(triton_block_m),
            )
        else:
            row_lse: torch.Tensor | None = None
            for start in range(0, target.shape[0], chunk):
                current_target = target[start : start + chunk]
                cost = 0.5 * torch.cdist(residual, current_target).square() / float(cost_scale)
                current = torch.logsumexp(
                    v[start : start + len(current_target)][None, :]
                    - cost / float(epsilon),
                    dim=1,
                )
                row_lse = current if row_lse is None else torch.logaddexp(row_lse, current)
            assert row_lse is not None
        return float(torch.max(torch.abs(u + row_lse - log_a)).cpu())

    u = init_u
    v = init_v
    iterations = 0
    error = math.inf
    converged = False
    if tolerance is None:
        u, v = solve_batch(minimum, u, v)
        iterations = minimum
        error = row_log_marginal_error(u, v)
        converged = True
    else:
        while iterations < maximum:
            batch = min(check, maximum - iterations)
            u, v = solve_batch(batch, u, v)
            iterations += batch
            if iterations < minimum:
                continue
            error = row_log_marginal_error(u, v)
            if error <= float(tolerance):
                converged = True
                break
    assert u is not None and v is not None
    if require_convergence and not converged:
        raise RuntimeError(
            "PNW inference dual did not converge: "
            f"row_log_marginal_error={error:.3e}, tolerance={tolerance}, "
            f"iterations={iterations}/{maximum}"
        )
    diagnostics = {
        "dual_solver": "sinkhorn_fixed_point",
        "dual_backend": chosen_backend,
        "dual_initialization": "provided" if init_v is not None else "zeros",
        "dual_iterations": int(iterations),
        "dual_minimum_iterations": int(minimum),
        "dual_max_iterations": int(maximum),
        "dual_check_every": int(check),
        "dual_tolerance": None if tolerance is None else float(tolerance),
        "dual_row_log_marginal_error": float(error),
        "dual_converged": bool(converged),
    }
    return u, v, diagnostics


@torch.no_grad()
def pnw_rank_values_and_average_jacobian(
    residual: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    sinkhorn_iters: int,
    cost_scale: float,
    chunk_size: int | None = 8192,
    dual_v: torch.Tensor | None = None,
    dual_u: torch.Tensor | None = None,
    dual_init_v: torch.Tensor | None = None,
    dual_tolerance: float | None = 1e-5,
    dual_max_iters: int = 200,
    dual_check_every: int = 5,
    dual_backend: str = "auto",
    dual_triton_block_m: int = 1024,
    require_dual_convergence: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Evaluate PNW ranks and the average fixed-potential map Jacobian.

    For the map returned by the production implementation,
    ``T(r) = softmax(v - ||r-z||^2/(2*epsilon*cost_scale)) @ z``.  Therefore
    its Jacobian is the conditional covariance of ``z`` divided by
    ``epsilon * cost_scale``.  The target-reference normalization is already
    encoded in the fitted dual ``v`` and introduces no additional factor.

    The target cloud is streamed in chunks, so memory is ``O(n*chunk_size)``
    rather than ``O(n*m_ref)``.  ``dual_v`` may be supplied for finite-
    difference checks that must hold the fitted OT potential fixed.
    """

    if residual.ndim != 2 or target.ndim != 2 or residual.shape[1] != target.shape[1]:
        raise ValueError("residual and target must be two-dimensional with equal width")
    if epsilon <= 0.0 or cost_scale <= 0.0:
        raise ValueError("epsilon and cost_scale must be positive")
    n, d = residual.shape
    m = target.shape[0]
    if n < 1 or m < 1:
        raise ValueError("residual and target must be nonempty")
    chunk = m if chunk_size is None or int(chunk_size) <= 0 else min(m, int(chunk_size))
    if dual_v is None:
        _, fitted_v, dual_diagnostics = reconstruct_pnw_dual(
            residual,
            target,
            epsilon=float(epsilon),
            cost_scale=float(cost_scale),
            minimum_iterations=int(sinkhorn_iters),
            tolerance=dual_tolerance,
            max_iterations=int(dual_max_iters),
            check_every=int(dual_check_every),
            backend=dual_backend,
            chunk_size=chunk,
            triton_block_m=int(dual_triton_block_m),
            init_u=dual_u,
            init_v=dual_init_v,
            require_convergence=bool(require_dual_convergence),
        )
    else:
        fitted_v = dual_v.to(device=target.device, dtype=target.dtype).reshape(-1)
        if len(fitted_v) != m:
            raise ValueError("dual_v must have one value per target point")
        dual_diagnostics = {
            "dual_solver": "supplied_fixed_potential",
            "dual_backend": "supplied",
            "dual_initialization": "provided",
            "dual_iterations": 0,
            "dual_minimum_iterations": 0,
            "dual_max_iterations": 0,
            "dual_check_every": 0,
            "dual_tolerance": None,
            "dual_row_log_marginal_error": math.nan,
            "dual_converged": None,
        }

    row_log_normalizer: torch.Tensor | None = None
    for start in range(0, m, chunk):
        current_target = target[start : start + chunk]
        cost = 0.5 * torch.cdist(residual, current_target).square() / float(cost_scale)
        logits = fitted_v[start : start + len(current_target)][None, :] - cost / float(epsilon)
        current_lse = torch.logsumexp(logits, dim=1)
        row_log_normalizer = (
            current_lse
            if row_log_normalizer is None
            else torch.logaddexp(row_log_normalizer, current_lse)
        )
    assert row_log_normalizer is not None

    ranks = torch.zeros((n, d), device=residual.device, dtype=residual.dtype)
    # MPS does not support float64 tensors. Keep the large probability and
    # matrix products on the accelerator in float32, but accumulate the small
    # d-by-d second moments on CPU in float64 for the structured Hessian.
    mps_cpu_accumulation = residual.device.type == "mps"
    second_moment = torch.zeros(
        (d, d),
        device="cpu" if mps_cpu_accumulation else residual.device,
        dtype=torch.float64,
    )
    row_mass = torch.zeros((n,), device=residual.device, dtype=residual.dtype)
    for start in range(0, m, chunk):
        current_target = target[start : start + chunk]
        cost = 0.5 * torch.cdist(residual, current_target).square() / float(cost_scale)
        logits = fitted_v[start : start + len(current_target)][None, :] - cost / float(epsilon)
        probability = torch.exp(logits - row_log_normalizer[:, None])
        ranks.add_(probability @ current_target)
        row_mass.add_(probability.sum(dim=1))
        if mps_cpu_accumulation:
            target_mass = probability.sum(dim=0) / float(n)
            current_second_moment = current_target.T @ (
                target_mass[:, None] * current_target
            )
            second_moment.add_(current_second_moment.cpu().double())
        else:
            target_mass = probability.sum(dim=0).double() / float(n)
            current64 = current_target.double()
            second_moment.add_(current64.T @ (target_mass[:, None] * current64))
    if mps_cpu_accumulation:
        rank_second_moment = (ranks.T @ ranks / float(n)).cpu().double()
    else:
        rank_second_moment = ranks.double().T @ ranks.double() / float(n)
    average_jacobian = (second_moment - rank_second_moment) / (
        float(epsilon) * float(cost_scale)
    )
    average_jacobian = 0.5 * (average_jacobian + average_jacobian.T)
    eigenvalues = torch.linalg.eigvalsh(average_jacobian)
    diagnostics = {
        "pnw_jacobian_scale": "conditional_target_covariance/(epsilon*cost_scale)",
        "reference_normalization_extra_factor": 1.0,
        "row_probability_error": float(torch.abs(row_mass - 1.0).max().cpu()),
        "jacobian_minimum_eigenvalue": float(eigenvalues[0].cpu()),
        "jacobian_maximum_eigenvalue": float(eigenvalues[-1].cpu()),
        "jacobian_condition_number": float(torch.linalg.cond(average_jacobian).cpu()),
        "chunk_size": int(chunk),
        "jacobian_accumulation_device": (
            "cpu_float64_from_mps_float32_chunks"
            if mps_cpu_accumulation
            else f"{residual.device.type}_float64"
        ),
        "dual_recomputed": dual_v is None,
        **dual_diagnostics,
    }
    return ranks, average_jacobian, diagnostics


def _plugin_linear_components(
    beta: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    sinkhorn_iters: int,
    cost_scale: float,
    l2_penalty: float,
    intercept_mode: str,
    x_mean: torch.Tensor | None,
    y_mean: torch.Tensor | None,
    derivative_ridge: float,
    chunk_size: int | None,
    dual_tolerance: float | None,
    dual_max_iters: int,
    dual_check_every: int,
    dual_backend: str,
    dual_triton_block_m: int,
    require_dual_convergence: bool,
    dual_u: torch.Tensor | None,
    dual_v: torch.Tensor | None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    design, residual = _uniform_design_and_residual(
        beta,
        x,
        y,
        intercept_mode=intercept_mode,
        x_mean=x_mean,
        y_mean=y_mean,
    )
    ranks, average_jacobian, map_diagnostics = pnw_rank_values_and_average_jacobian(
        residual.detach(),
        target.detach(),
        epsilon=epsilon,
        sinkhorn_iters=sinkhorn_iters,
        cost_scale=cost_scale,
        chunk_size=chunk_size,
        dual_u=dual_u,
        dual_v=dual_v,
        dual_tolerance=dual_tolerance,
        dual_max_iters=dual_max_iters,
        dual_check_every=dual_check_every,
        dual_backend=dual_backend,
        dual_triton_block_m=dual_triton_block_m,
        require_dual_convergence=require_dual_convergence,
    )
    n, p = design.shape
    d = residual.shape[1]
    if design.device.type == "mps":
        design64 = design.cpu().double()
        design_second_moment = design64.T @ design64 / float(n)
    else:
        design_second_moment = design.double().T @ design.double() / float(n)
    # Under standard column-major vec(B), vec(J B Sigma_X) is represented by
    # Sigma_X tensor J.
    hessian = np.kron(
        design_second_moment.cpu().numpy(),
        average_jacobian.cpu().numpy(),
    )
    if l2_penalty:
        hessian = hessian + float(l2_penalty) * np.eye(d * p)
    case_scores = vec_coefficient_batch(
        torch.einsum("nd,np->ndp", ranks, design)
    )
    case_scores_np = case_scores.cpu().double().numpy()
    influence = influence_from_scores_and_hessian(
        hessian,
        case_scores_np,
        ridge=derivative_ridge,
    )
    symmetric_hessian = 0.5 * (hessian + hessian.T)
    hessian_eigenvalues = np.linalg.eigvalsh(symmetric_hessian)
    sigma_eigenvalues = np.linalg.eigvalsh(design_second_moment.cpu().numpy())
    diagnostics = {
        "parameter_dimension": int(d * p),
        "case_count": int(n),
        "plugin_score_mean_norm": float(np.linalg.norm(case_scores_np.mean(axis=0))),
        "hessian_minimum_eigenvalue": float(hessian_eigenvalues[0]),
        "hessian_maximum_eigenvalue": float(hessian_eigenvalues[-1]),
        "hessian_condition_number": float(np.linalg.cond(symmetric_hessian)),
        "hessian_rank": int(np.linalg.matrix_rank(symmetric_hessian)),
        "design_minimum_eigenvalue": float(sigma_eigenvalues[0]),
        "design_maximum_eigenvalue": float(sigma_eigenvalues[-1]),
        "design_condition_number": float(np.linalg.cond(design_second_moment.cpu().numpy())),
        "influence_mean_norm": float(np.linalg.norm(influence.mean(axis=0))),
        "inference_method": "plugin_covariance",
        "inference_backend": "plugin_covariance_numpy_dense_linear_algebra",
        "covariance_backend": "numpy_dense_linear_algebra",
        "rank_jacobian_backend": "chunked_pytorch",
        "dual_solver_backend": map_diagnostics["dual_backend"],
        "complete_candidate_dependent_hessian": False,
        "population_hessian_plugin": True,
        "vectorization": COEFFICIENT_VECTORIZATION,
        "hessian_vectorization": (
            "column_major:kron(design_second_moment,average_map_jacobian)"
        ),
        "frozen_rank_covariance_meat": True,
        "conditional_on_fixed_reference": True,
        "m_ref": int(target.shape[0]),
        "epsilon": float(epsilon),
        "sinkhorn_iters": int(sinkhorn_iters),
        "cost_scale": float(cost_scale),
        "l2_penalty": float(l2_penalty),
        "intercept_mode": intercept_mode,
        "fixed_pilot_means": intercept_mode.lower() == "pilot",
        **map_diagnostics,
    }
    components = {
        "hessian": symmetric_hessian,
        "case_scores": case_scores_np,
        "centered_case_scores": case_scores_np - case_scores_np.mean(axis=0, keepdims=True),
        "design_second_moment": design_second_moment.cpu().numpy(),
        "average_map_jacobian": average_jacobian.cpu().numpy(),
        "ranks": ranks.cpu().numpy(),
        "residuals": residual.detach().cpu().numpy(),
    }
    return influence, diagnostics, components


def plugin_linear_influence(
    beta: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    sinkhorn_iters: int,
    cost_scale: float,
    l2_penalty: float = 0.0,
    intercept_mode: str = "zero",
    x_mean: torch.Tensor | None = None,
    y_mean: torch.Tensor | None = None,
    derivative_ridge: float = 0.0,
    chunk_size: int | None = 8192,
    dual_tolerance: float | None = 1e-5,
    dual_max_iters: int = 200,
    dual_check_every: int = 5,
    dual_backend: str = "auto",
    dual_triton_block_m: int = 1024,
    require_dual_convergence: bool = True,
    dual_u: torch.Tensor | None = None,
    dual_v: torch.Tensor | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the OT-rank plug-in influence function for one reference cloud."""

    influence, diagnostics, _ = _plugin_linear_components(
        beta,
        x,
        y,
        target,
        epsilon=epsilon,
        sinkhorn_iters=sinkhorn_iters,
        cost_scale=cost_scale,
        l2_penalty=l2_penalty,
        intercept_mode=intercept_mode,
        x_mean=x_mean,
        y_mean=y_mean,
        derivative_ridge=derivative_ridge,
        chunk_size=chunk_size,
        dual_tolerance=dual_tolerance,
        dual_max_iters=dual_max_iters,
        dual_check_every=dual_check_every,
        dual_backend=dual_backend,
        dual_triton_block_m=dual_triton_block_m,
        require_dual_convergence=require_dual_convergence,
        dual_u=dual_u,
        dual_v=dual_v,
    )
    return influence, diagnostics


def plugin_linear_components(
    beta: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    target: torch.Tensor,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Return plug-in IF together with Hessian, scores, ranks, and residuals."""

    defaults = {
        "l2_penalty": 0.0,
        "intercept_mode": "zero",
        "x_mean": None,
        "y_mean": None,
        "derivative_ridge": 0.0,
        "chunk_size": 8192,
        "dual_tolerance": 1e-5,
        "dual_max_iters": 200,
        "dual_check_every": 5,
        "dual_backend": "auto",
        "dual_triton_block_m": 1024,
        "require_dual_convergence": True,
        "dual_u": None,
        "dual_v": None,
    }
    defaults.update(kwargs)
    return _plugin_linear_components(beta, x, y, target, **defaults)


def plugin_covariance(
    coefficients: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    cost_scales: Sequence[float],
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epsilon: float,
    sinkhorn_iters: int,
    l2_penalty: float = 0.0,
    intercept_mode: str = "zero",
    x_mean: torch.Tensor | None = None,
    y_mean: torch.Tensor | None = None,
    derivative_ridge: float = 0.0,
    chunk_size: int | None = 8192,
    dual_tolerance: float | None = 1e-5,
    dual_max_iters: int = 200,
    dual_check_every: int = 5,
    dual_backend: str = "auto",
    dual_triton_block_m: int = 1024,
    require_dual_convergence: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Average plug-in reference-wise IFs before forming covariance."""

    if not (
        len(coefficients) == len(targets) == len(cost_scales) and coefficients
    ):
        raise ValueError("coefficients, targets, and cost_scales must have equal nonzero length")
    influences = []
    diagnostics = []
    for beta, target, scale in zip(coefficients, targets, cost_scales):
        current, detail = plugin_linear_influence(
            beta,
            x,
            y,
            target,
            epsilon=epsilon,
            sinkhorn_iters=sinkhorn_iters,
            cost_scale=float(scale),
            l2_penalty=l2_penalty,
            intercept_mode=intercept_mode,
            x_mean=x_mean,
            y_mean=y_mean,
            derivative_ridge=derivative_ridge,
            chunk_size=chunk_size,
            dual_tolerance=dual_tolerance,
            dual_max_iters=dual_max_iters,
            dual_check_every=dual_check_every,
            dual_backend=dual_backend,
            dual_triton_block_m=dual_triton_block_m,
            require_dual_convergence=require_dual_convergence,
        )
        influences.append(current)
        diagnostics.append(detail)
    stacked = np.stack(influences)
    ensemble_influence = stacked.mean(axis=0)
    covariance = ensemble_influence_covariance(stacked)
    return ensemble_influence, covariance, diagnostics


def case_weight_linear_influence(
    beta: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    sinkhorn_iters: int,
    cost_scale: float,
    l2_penalty: float = 0.0,
    intercept_mode: str = "zero",
    x_mean: torch.Tensor | None = None,
    y_mean: torch.Tensor | None = None,
    derivative_ridge: float = 0.0,
    vectorize_derivatives: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a case-weight local-root IF for one fixed reference cloud."""

    weights = torch.full(
        (len(x),),
        1.0 / len(x),
        dtype=beta.dtype,
        device=beta.device,
    )

    def objective(
        current_beta: torch.Tensor,
        current_weights: torch.Tensor,
    ) -> torch.Tensor:
        return weighted_dense_pnw_objective(
            current_beta,
            current_weights,
            x,
            y,
            target,
            epsilon=epsilon,
            sinkhorn_iters=sinkhorn_iters,
            cost_scale=cost_scale,
            l2_penalty=l2_penalty,
            intercept_mode=intercept_mode,
            x_mean=x_mean,
            y_mean=y_mean,
        )

    influence, diagnostics = autograd_case_weight_influence(
        objective,
        beta,
        weights,
        ridge=derivative_ridge,
        vectorize_derivatives=vectorize_derivatives,
    )
    diagnostics.update(
        {
            "inference_method": "case_weight_if_covariance",
            "inference_backend": "case_weight_if_dense_pytorch_autograd",
            "fit_backend_may_differ": True,
            "conditional_on_fixed_reference": True,
            "m_ref": int(target.shape[0]),
            "epsilon": float(epsilon),
            "sinkhorn_iters": int(sinkhorn_iters),
            "cost_scale": float(cost_scale),
            "l2_penalty": float(l2_penalty),
            "intercept_mode": intercept_mode,
            "fixed_pilot_means": intercept_mode.lower() == "pilot",
        }
    )
    return influence, diagnostics


def case_weight_score_derivatives(
    beta: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    sinkhorn_iters: int,
    cost_scale: float,
    l2_penalty: float = 0.0,
    intercept_mode: str = "zero",
    x_mean: torch.Tensor | None = None,
    y_mean: torch.Tensor | None = None,
    vectorize_derivatives: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return full Hessian, case-weight ``D=-C``, and plug-in score ``P``.

    Both score arrays have shape ``(n, q)`` and are centered across cases.
    This interface is intended for finite-sample feedback diagnostics; it does
    not replace either covariance estimator.
    """

    weights = torch.full(
        (len(x),),
        1.0 / len(x),
        dtype=beta.dtype,
        device=beta.device,
    )

    def objective(
        current_beta: torch.Tensor,
        current_weights: torch.Tensor,
    ) -> torch.Tensor:
        return weighted_dense_pnw_objective(
            current_beta,
            current_weights,
            x,
            y,
            target,
            epsilon=epsilon,
            sinkhorn_iters=sinkhorn_iters,
            cost_scale=cost_scale,
            l2_penalty=l2_penalty,
            intercept_mode=intercept_mode,
            x_mean=x_mean,
            y_mean=y_mean,
        )

    hessian, contamination_derivative, diagnostics = (
        autograd_case_weight_score_derivatives(
            objective,
            beta,
            weights,
            vectorize_derivatives=vectorize_derivatives,
        )
    )
    theta = beta.detach()
    _, _, ranks, design = _weighted_dense_pnw_state(
        theta,
        weights,
        x,
        y,
        target,
        epsilon=epsilon,
        sinkhorn_iters=sinkhorn_iters,
        cost_scale=cost_scale,
        intercept_mode=intercept_mode,
        x_mean=x_mean,
        y_mean=y_mean,
    )
    plugin_score = vec_coefficient_batch(
        torch.einsum("nd,np->ndp", ranks, design)
    )
    plugin_score_np = plugin_score.detach().cpu().double().numpy()
    plugin_score_np -= plugin_score_np.mean(axis=0, keepdims=True)
    case_weight_score = -contamination_derivative.T
    case_weight_score -= case_weight_score.mean(axis=0, keepdims=True)
    diagnostics.update(
        {
            "score_sign_convention": "D=-C; plugin P=positive rank outer design",
            "vectorization": COEFFICIENT_VECTORIZATION,
            "plugin_score_mean_norm": float(np.linalg.norm(plugin_score_np.mean(axis=0))),
            "case_weight_score_mean_norm": float(
                np.linalg.norm(case_weight_score.mean(axis=0))
            ),
            "conditional_on_fixed_reference": True,
            "m_ref": int(target.shape[0]),
            "epsilon": float(epsilon),
            "sinkhorn_iters": int(sinkhorn_iters),
            "cost_scale": float(cost_scale),
        }
    )
    return hessian, case_weight_score, plugin_score_np, diagnostics


def case_weight_if_covariance(
    coefficients: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    cost_scales: Sequence[float],
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epsilon: float,
    sinkhorn_iters: int,
    l2_penalty: float = 0.0,
    intercept_mode: str = "zero",
    x_mean: torch.Tensor | None = None,
    y_mean: torch.Tensor | None = None,
    derivative_ridge: float = 0.0,
    vectorize_derivatives: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Average case-weight reference-wise IFs, then form covariance."""

    if not (
        len(coefficients) == len(targets) == len(cost_scales) and coefficients
    ):
        raise ValueError("coefficients, targets, and cost_scales must have equal nonzero length")
    influences = []
    diagnostics = []
    for beta, target, scale in zip(coefficients, targets, cost_scales):
        current, detail = case_weight_linear_influence(
            beta,
            x,
            y,
            target,
            epsilon=epsilon,
            sinkhorn_iters=sinkhorn_iters,
            cost_scale=float(scale),
            l2_penalty=l2_penalty,
            intercept_mode=intercept_mode,
            x_mean=x_mean,
            y_mean=y_mean,
            derivative_ridge=derivative_ridge,
            vectorize_derivatives=vectorize_derivatives,
        )
        influences.append(current)
        diagnostics.append(detail)
    stacked = np.stack(influences)
    ensemble_influence = stacked.mean(axis=0)
    covariance = ensemble_influence_covariance(stacked)
    return ensemble_influence, covariance, diagnostics
