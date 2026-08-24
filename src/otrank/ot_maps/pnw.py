"""Pooladian--Niles-Weed-style entropic OT score utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .base import FittedOTMap
from .reference import sample_reference


@dataclass
class PNWEntropicOTConfig:
    """Configuration for direct entropic OT score estimation."""

    ot_estimator: str = "pnw"
    m_ref: int = 16384
    epsilon: float = 0.10
    sinkhorn_iters: int = 1
    reference_distribution: str = "uniform_ball"
    reference_radius: float = 1.0
    reference_values: Any | None = None
    seed: int = 20260721
    cost_scale: float | None = None
    topk_barycentric: int | None = None
    chunk_size: int | None = None
    detach_sinkhorn_gradient: bool = False
    backend: str = "dense_triton"
    implicit_solve_dtype: str = "float32"
    implicit_gmres_iters: int = 30
    implicit_gmres_tol: float = 1e-6
    implicit_ridge: float = 1e-8
    implicit_chunk_size: int = 1024
    implicit_cache_dtype: str = "float32"
    sinkhorn_warm_start: bool = True
    triton_direct_block_n: int = 8
    triton_direct_block_m: int = 1024
    triton_matvec_block_n: int = 32
    triton_matvec_block_m: int = 512
    triton_implicit_add_block_m: int = 512
    triton_num_warps: int = 4


_SINKHORN_WARM_START_CACHE: dict[tuple[int, int, tuple[int, ...], tuple[int, ...], str, str], tuple[torch.Tensor, torch.Tensor]] = {}

OT_ESTIMATOR_PNW = "pnw"

DENSE_PYTORCH_BACKEND = "dense_pytorch"
DENSE_TRITON_BACKEND = "dense_triton"

SUPPORTED_OT_ESTIMATORS = {OT_ESTIMATOR_PNW}
SUPPORTED_OT_BACKENDS = {DENSE_PYTORCH_BACKEND, DENSE_TRITON_BACKEND}

_LEGACY_BACKEND_ALIASES = {
    "torch": (OT_ESTIMATOR_PNW, DENSE_PYTORCH_BACKEND),
    "implicit_triton_loss_cached_ab_gemm_add": (OT_ESTIMATOR_PNW, DENSE_TRITON_BACKEND),
}


def normalize_ot_estimator_backend(ot_estimator: str = OT_ESTIMATOR_PNW, backend: str = DENSE_TRITON_BACKEND) -> tuple[str, str]:
    """Normalize the public dense PNW estimator/backend pair."""

    estimator = str(ot_estimator or OT_ESTIMATOR_PNW).lower()
    chosen_backend = str(backend or DENSE_TRITON_BACKEND).lower()
    if chosen_backend in _LEGACY_BACKEND_ALIASES:
        legacy_estimator, legacy_backend = _LEGACY_BACKEND_ALIASES[chosen_backend]
        if estimator not in {"", OT_ESTIMATOR_PNW, legacy_estimator}:
            raise ValueError(
                f"legacy backend {chosen_backend!r} implies ot_estimator={legacy_estimator!r}, "
                f"not {estimator!r}"
            )
        estimator, chosen_backend = legacy_estimator, legacy_backend
    if estimator != OT_ESTIMATOR_PNW:
        raise ValueError("only the dense PNW OT estimator is active in the main code path")
    if chosen_backend not in SUPPORTED_OT_BACKENDS:
        raise ValueError("backend must be 'dense_pytorch' or 'dense_triton'")
    return estimator, chosen_backend


def clear_pnw_sinkhorn_warm_start(warm_start_id: int) -> None:
    """Clear cached Sinkhorn duals for a profile PNW fit."""

    key_prefix = int(warm_start_id)
    for key in list(_SINKHORN_WARM_START_CACHE):
        if key[0] == key_prefix:
            del _SINKHORN_WARM_START_CACHE[key]


def median_cost_scale(source: torch.Tensor, target: torch.Tensor) -> float:
    """Robust fixed scale for squared Euclidean transport costs."""

    with torch.no_grad():
        cost = 0.5 * torch.cdist(source.detach(), target.detach()).pow(2)
        scale = torch.median(cost)
        if torch.isfinite(scale) and float(scale.item()) > 1e-12:
            return float(scale.item())
    return 1.0


def sinkhorn_log_scalings(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    sinkhorn_iters: int,
    cost_scale: float,
    chunk_size: int | None = None,
    init_u: torch.Tensor | None = None,
    init_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Log-domain Sinkhorn scalings with gradients through ``source``."""

    n = source.shape[0]
    m = target.shape[0]
    dtype = source.dtype
    device = source.device
    eps = float(epsilon)
    log_a = torch.full((n,), -torch.log(torch.tensor(float(n), device=device, dtype=dtype)), device=device, dtype=dtype)
    log_b = torch.full((m,), -torch.log(torch.tensor(float(m), device=device, dtype=dtype)), device=device, dtype=dtype)
    if init_u is not None and tuple(init_u.shape) == tuple(log_a.shape):
        u = init_u.to(device=device, dtype=dtype).detach().clone()
    else:
        u = torch.zeros_like(log_a)
    if init_v is not None and tuple(init_v.shape) == tuple(log_b.shape):
        v = init_v.to(device=device, dtype=dtype).detach().clone()
    else:
        v = torch.zeros_like(log_b)
    chunk = 0 if chunk_size is None else int(chunk_size)
    if chunk > 0 and chunk < m:
        for _ in range(max(1, int(sinkhorn_iters))):
            row_lse = None
            for start in range(0, m, chunk):
                target_chunk = target[start : start + chunk]
                cost_chunk = 0.5 * torch.cdist(source, target_chunk).pow(2) / float(cost_scale)
                value = -cost_chunk / eps + v[start : start + target_chunk.shape[0]][None, :]
                chunk_lse = torch.logsumexp(value, dim=1)
                row_lse = chunk_lse if row_lse is None else torch.logaddexp(row_lse, chunk_lse)
            u = log_a - row_lse

            v_chunks = []
            for start in range(0, m, chunk):
                target_chunk = target[start : start + chunk]
                cost_chunk = 0.5 * torch.cdist(source, target_chunk).pow(2) / float(cost_scale)
                value = -cost_chunk / eps + u[:, None]
                v_chunks.append(log_b[start : start + target_chunk.shape[0]] - torch.logsumexp(value, dim=0))
            v = torch.cat(v_chunks, dim=0)
        # The dense cost matrix is intentionally not retained on the chunked path.
        return u, v, torch.empty(0, device=device, dtype=dtype)

    cost = 0.5 * torch.cdist(source, target).pow(2) / float(cost_scale)
    log_k = -cost / eps
    for _ in range(max(1, int(sinkhorn_iters))):
        u = log_a - torch.logsumexp(log_k + v[None, :], dim=1)
        v = log_b - torch.logsumexp(log_k + u[:, None], dim=0)
    return u, v, cost


def _keops_lazy_tensors(source: torch.Tensor, target: torch.Tensor, values_j: torch.Tensor | None = None):
    _configure_keops_cuda_from_conda()
    from pykeops.torch import LazyTensor

    source_i = LazyTensor(source[:, None, :])
    target_j = LazyTensor(target[None, :, :])
    values = None if values_j is None else LazyTensor(values_j[None, :, None])
    return source_i, target_j, values


def _keops_logits(
    source: torch.Tensor,
    target: torch.Tensor,
    values_j: torch.Tensor,
    *,
    epsilon: float,
    cost_scale: float,
):
    source_i, target_j, values = _keops_lazy_tensors(source, target, values_j)
    cost = 0.5 * ((source_i - target_j) ** 2).sum(-1) / float(cost_scale)
    return values - cost / float(epsilon), target_j


def sinkhorn_log_scalings_keops(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    sinkhorn_iters: int,
    cost_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-domain Sinkhorn scalings using KeOps pairwise reductions."""

    n = source.shape[0]
    m = target.shape[0]
    dtype = source.dtype
    device = source.device
    log_a = torch.full((n,), -torch.log(torch.tensor(float(n), device=device, dtype=dtype)), device=device, dtype=dtype)
    log_b = torch.full((m,), -torch.log(torch.tensor(float(m), device=device, dtype=dtype)), device=device, dtype=dtype)
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(max(1, int(sinkhorn_iters))):
        logits_row, _ = _keops_logits(source, target, v, epsilon=epsilon, cost_scale=cost_scale)
        u = log_a - logits_row.logsumexp(dim=1).squeeze(-1)

        source_i, target_j, _ = _keops_lazy_tensors(source, target)
        from pykeops.torch import LazyTensor

        u_i = LazyTensor(u[:, None, None])
        cost = 0.5 * ((source_i - target_j) ** 2).sum(-1) / float(cost_scale)
        logits_col = u_i - cost / float(epsilon)
        v = log_b - logits_col.logsumexp(dim=0).squeeze(-1)
    return u, v


def sinkhorn_log_scalings_triton(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    sinkhorn_iters: int,
    cost_scale: float,
    block_m: int = 1024,
    init_u: torch.Tensor | None = None,
    init_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-domain Sinkhorn scalings using Triton row-logsumexp reductions.

    This experimental backend keeps gradients through both row and column
    reductions.  It only replaces the dual-scaling reductions; the final
    barycentric score computation is handled by the caller.
    """

    from .pnw_triton import triton_row_logsumexp

    n = source.shape[0]
    m = target.shape[0]
    dtype = source.dtype
    device = source.device
    log_a = torch.full((n,), -torch.log(torch.tensor(float(n), device=device, dtype=dtype)), device=device, dtype=dtype)
    log_b = torch.full((m,), -torch.log(torch.tensor(float(m), device=device, dtype=dtype)), device=device, dtype=dtype)
    if init_u is not None and tuple(init_u.shape) == tuple(log_a.shape):
        u = init_u.to(device=device, dtype=dtype).detach().clone()
    else:
        u = torch.zeros_like(log_a)
    if init_v is not None and tuple(init_v.shape) == tuple(log_b.shape):
        v = init_v.to(device=device, dtype=dtype).detach().clone()
    else:
        v = torch.zeros_like(log_b)
    for _ in range(max(1, int(sinkhorn_iters))):
        row_lse = triton_row_logsumexp(
            source,
            target,
            v,
            epsilon=float(epsilon),
            cost_scale=float(cost_scale),
            block_m=int(block_m),
        )
        u = log_a - row_lse

        col_lse = triton_row_logsumexp(
            target,
            source,
            u,
            epsilon=float(epsilon),
            cost_scale=float(cost_scale),
            block_m=int(block_m),
        )
        v = log_b - col_lse
    return u, v


def _project_zero_mean(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean()


def _gmres_matrix_free(
    matvec,
    rhs: torch.Tensor,
    *,
    max_iter: int,
    tol: float,
    matvec_returns_projected: bool = False,
    check_every: int = 5,
) -> torch.Tensor:
    """Small fixed-iteration GMRES solver for the implicit Sinkhorn backend.

    This variant intentionally avoids host synchronization in the hot loop.  It
    does not recompute the full residual with an additional matvec at every
    iteration; the final iterate is obtained from the small Arnoldi least-squares
    problem after ``max_iter`` steps.  The ``tol`` argument is kept for API
    compatibility with earlier checkpoints/configs.
    """

    iters = max(1, int(max_iter))
    b = _project_zero_mean(rhs.detach())
    beta = torch.linalg.norm(b)
    beta_safe = beta.clamp_min(torch.finfo(b.dtype).eps)
    q_basis = torch.empty((iters, b.numel()), device=b.device, dtype=b.dtype)
    q_basis[0] = b / beta_safe
    h = torch.zeros((iters + 1, iters), device=b.device, dtype=b.dtype)
    rhs_small = torch.zeros((iters + 1,), device=b.device, dtype=b.dtype)
    rhs_small[0] = beta
    eps = torch.finfo(b.dtype).eps
    active_iters = iters
    check = max(0, int(check_every))
    for k in range(iters):
        w = matvec(q_basis[k])
        if not matvec_returns_projected:
            w = _project_zero_mean(w)
        q_view = q_basis[: k + 1]
        coeff = q_view @ w
        h[: k + 1, k] = coeff
        w = w - coeff @ q_view
        h[k + 1, k] = torch.linalg.norm(w)
        if k + 1 < iters:
            q_basis[k + 1] = w / h[k + 1, k].clamp_min(eps)
        if check and float(tol) > 0.0 and ((k + 1) % check == 0 or k + 1 == iters):
            h_small = h[: k + 2, : k + 1]
            rhs_check = rhs_small[: k + 2]
            y_check = torch.linalg.lstsq(h_small, rhs_check[:, None]).solution[:, 0]
            rel = torch.linalg.norm(h_small @ y_check - rhs_check) / beta_safe
            if float(rel.item()) <= float(tol):
                active_iters = k + 1
                break
    h_final = h[: active_iters + 1, :active_iters]
    rhs_final = rhs_small[: active_iters + 1]
    y = torch.linalg.lstsq(h_final, rhs_final[:, None]).solution[:, 0]
    out = y @ q_basis[:active_iters]
    return out if matvec_returns_projected else _project_zero_mean(out)


def _gmres_matrix_free_legacy(
    matvec,
    rhs: torch.Tensor,
    *,
    max_iter: int,
    tol: float,
) -> torch.Tensor:
    """Legacy GMRES path retained for parity benchmarks.

    This is the pre-optimization implementation: it synchronizes on scalar
    convergence checks and recomputes a full residual matvec at each iteration.
    """

    b = _project_zero_mean(rhs.detach())
    beta = torch.linalg.norm(b)
    if float(beta.item()) == 0.0:
        return torch.zeros_like(b)
    q_vectors = [b / beta]
    h = torch.zeros((max_iter + 1, max_iter), device=b.device, dtype=b.dtype)
    rhs_small_full = torch.zeros((max_iter + 1,), device=b.device, dtype=b.dtype)
    rhs_small_full[0] = beta
    best_x = torch.zeros_like(b)
    for k in range(max(1, int(max_iter))):
        w = _project_zero_mean(matvec(q_vectors[k]))
        for j in range(k + 1):
            h[j, k] = torch.dot(q_vectors[j], w)
            w = w - h[j, k] * q_vectors[j]
        h[k + 1, k] = torch.linalg.norm(w)
        if float(h[k + 1, k].item()) > 1e-14:
            q_vectors.append(w / h[k + 1, k])
        h_small = h[: k + 2, : k + 1]
        rhs_small = rhs_small_full[: k + 2]
        y = torch.linalg.lstsq(h_small, rhs_small[:, None]).solution[:, 0]
        q_mat = torch.stack(q_vectors[: k + 1], dim=1)
        best_x = _project_zero_mean(q_mat @ y)
        rel = float(torch.linalg.norm(_project_zero_mean(matvec(best_x)) - b).item() / max(1e-12, float(beta.item())))
        if rel <= float(tol) or float(h[k + 1, k].item()) <= 1e-14:
            break
    return best_x


def _gemm_implicit_add_cached_ab(
    residuals: torch.Tensor,
    target: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    lam_u: torch.Tensor,
    lam_v: torch.Tensor,
    *,
    epsilon: float,
    cost_scale: float,
) -> torch.Tensor:
    """Packed-GEMM implicit correction for cached A/B PNW backward.

    Computes the same correction as
    ``sum_j (lam_u_i A_ij + B_ij lam_v_j) * (res_i - target_j)``, but avoids
    rereading ``A`` and ``B`` once for each response dimension.
    """

    solve_dtype = residuals.dtype
    target_work = target.to(dtype=solve_dtype)
    a_work = a.to(dtype=solve_dtype)
    b_work = b.to(dtype=solve_dtype)
    lam_u_work = lam_u.to(dtype=solve_dtype)
    lam_v_work = lam_v.to(dtype=solve_dtype)

    ones_target = torch.cat(
        (
            torch.ones((target_work.shape[0], 1), device=target_work.device, dtype=target_work.dtype),
            target_work,
        ),
        dim=1,
    )
    lam_target = torch.cat((lam_v_work[:, None], lam_v_work[:, None] * target_work), dim=1)
    a_packed = a_work @ ones_target
    b_packed = b_work @ lam_target
    coeff_sum = lam_u_work * a_packed[:, 0] + b_packed[:, 0]
    coeff_target = lam_u_work[:, None] * a_packed[:, 1:] + b_packed[:, 1:]
    return (residuals * coeff_sum[:, None] - coeff_target) / (float(epsilon) * float(cost_scale))


def _dtype_code_from_name(name: str) -> int:
    value = str(name).lower()
    if value in {"float32", "fp32"}:
        return 32
    if value in {"float16", "fp16", "half"}:
        return 16
    if value in {"bfloat16", "bf16"}:
        return 116
    raise ValueError("dtype name must be 'float32', 'float16', or 'bfloat16'")


def _dtype_from_code(code: int) -> torch.dtype:
    if int(code) == 32:
        return torch.float32
    if int(code) == 16:
        return torch.float16
    if int(code) == 116:
        return torch.bfloat16
    raise ValueError("dtype code must be 32, 16, or 116")


def _stream_b_times_q(
    source: torch.Tensor,
    target: torch.Tensor,
    u: torch.Tensor,
    q: torch.Tensor,
    *,
    epsilon: float,
    cost_scale: float,
    chunk_size: int,
) -> torch.Tensor:
    h = torch.zeros((source.shape[0],), device=source.device, dtype=source.dtype)
    chunk = max(1, int(chunk_size))
    for start in range(0, target.shape[0], chunk):
        target_chunk = target[start : start + chunk]
        cost = 0.5 * torch.cdist(source, target_chunk).pow(2) / float(cost_scale)
        weights = torch.softmax(u[:, None] - cost / float(epsilon), dim=0)
        h = h + weights @ q[start : start + target_chunk.shape[0]]
    return h


def _stream_at_times_h(
    source: torch.Tensor,
    target: torch.Tensor,
    v: torch.Tensor,
    row_lse: torch.Tensor,
    h: torch.Tensor,
    *,
    epsilon: float,
    cost_scale: float,
    chunk_size: int,
) -> torch.Tensor:
    out = torch.empty((target.shape[0],), device=source.device, dtype=source.dtype)
    chunk = max(1, int(chunk_size))
    for start in range(0, target.shape[0], chunk):
        target_chunk = target[start : start + chunk]
        cost = 0.5 * torch.cdist(source, target_chunk).pow(2) / float(cost_scale)
        logits = v[start : start + target_chunk.shape[0]][None, :] - cost / float(epsilon)
        weights = torch.exp(logits - row_lse[:, None])
        out[start : start + target_chunk.shape[0]] = weights.T @ h
    return out


def _stream_implicit_add(
    source: torch.Tensor,
    target: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    lam_v: torch.Tensor,
    *,
    epsilon: float,
    cost_scale: float,
    chunk_size: int,
) -> torch.Tensor:
    lam_u = -_stream_b_times_q(
        source,
        target,
        u,
        lam_v,
        epsilon=epsilon,
        cost_scale=cost_scale,
        chunk_size=chunk_size,
    )
    row_lse = torch.empty((source.shape[0],), device=source.device, dtype=source.dtype)
    chunk = max(1, int(chunk_size))
    row_lse_acc = None
    for start in range(0, target.shape[0], chunk):
        target_chunk = target[start : start + chunk]
        cost = 0.5 * torch.cdist(source, target_chunk).pow(2) / float(cost_scale)
        chunk_lse = torch.logsumexp(v[start : start + target_chunk.shape[0]][None, :] - cost / float(epsilon), dim=1)
        row_lse_acc = chunk_lse if row_lse_acc is None else torch.logaddexp(row_lse_acc, chunk_lse)
    row_lse.copy_(row_lse_acc)

    out = torch.zeros_like(source)
    for start in range(0, target.shape[0], chunk):
        target_chunk = target[start : start + chunk]
        cost = 0.5 * torch.cdist(source, target_chunk).pow(2) / float(cost_scale)
        a = torch.exp(v[start : start + target_chunk.shape[0]][None, :] - cost / float(epsilon) - row_lse[:, None])
        b = torch.softmax(u[:, None] - cost / float(epsilon), dim=0)
        coeff = lam_u[:, None] * a + b * lam_v[start : start + target_chunk.shape[0]][None, :]
        out = out + torch.sum(coeff[:, :, None] * (source[:, None, :] - target_chunk[None, :, :]), dim=1) / (
            float(epsilon) * float(cost_scale)
        )
    return out


def _sinkhorn_no_grad_lowmem(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    sinkhorn_iters: int,
    cost_scale: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        return sinkhorn_log_scalings(
            source,
            target,
            epsilon=epsilon,
            sinkhorn_iters=sinkhorn_iters,
            cost_scale=cost_scale,
            chunk_size=chunk_size,
        )[:2]


class _ImplicitCachedABPNWLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        residuals: torch.Tensor,
        target: torch.Tensor,
        epsilon: float,
        sinkhorn_iters: int,
        cost_scale: float,
        solve_dtype_code: int,
        cache_dtype_code: int,
        gmres_iters: int,
        gmres_tol: float,
        ridge: float,
        fused_code: int,
        direct_code: int,
        warm_start_id: int,
        direct_block_n: int,
        direct_block_m: int,
        matvec_block_n: int,
        matvec_block_m: int,
        implicit_add_block_m: int,
        triton_num_warps: int,
    ) -> torch.Tensor:
        if residuals.dtype != torch.float32 or target.dtype != torch.float32:
            raise TypeError("implicit_triton_loss_cached_ab currently requires float32 residuals and target")
        from .pnw_triton import triton_pnw_row_loss, triton_pnw_row_loss_tiled, triton_pnw_row_loss_tiled_state

        solve_dtype = torch.float64 if int(solve_dtype_code) == 64 else torch.float32
        work_res = residuals.detach().to(dtype=solve_dtype)
        work_target = target.detach().to(dtype=solve_dtype)
        warm_key = None
        init_u = None
        init_v = None
        if int(warm_start_id) > 0:
            warm_key = (
                int(warm_start_id),
                int(target.data_ptr()),
                tuple(work_res.shape),
                tuple(work_target.shape),
                str(work_res.device),
                str(work_res.dtype),
            )
            cached = _SINKHORN_WARM_START_CACHE.get(warm_key)
            if cached is not None:
                init_u, init_v = cached
        with torch.no_grad():
            u, v, _ = sinkhorn_log_scalings(
                work_res,
                work_target,
                epsilon=float(epsilon),
                sinkhorn_iters=int(sinkhorn_iters),
                cost_scale=float(cost_scale),
                chunk_size=None,
                init_u=init_u,
                init_v=init_v,
            )
        if warm_key is not None:
            _SINKHORN_WARM_START_CACHE[warm_key] = (u.detach(), v.detach())
        v_direct = v.to(dtype=torch.float32)
        if int(direct_code) == 1:
            row_loss, direct_lse = triton_pnw_row_loss_tiled_state(
                residuals,
                target,
                v_direct,
                epsilon=float(epsilon),
                cost_scale=float(cost_scale),
                block_n=int(direct_block_n),
                block_m=int(direct_block_m),
                num_warps=int(triton_num_warps),
            )
        else:
            row_loss = triton_pnw_row_loss(
                residuals,
                target,
                v_direct,
                epsilon=float(epsilon),
                cost_scale=float(cost_scale),
            )
            direct_lse = torch.empty((0,), device=residuals.device, dtype=residuals.dtype)
        ctx.save_for_backward(
            residuals.detach(),
            target.detach(),
            u.detach(),
            v.detach(),
            v_direct.detach(),
            direct_lse.detach(),
            row_loss.detach(),
        )
        ctx.epsilon = float(epsilon)
        ctx.sinkhorn_iters = int(sinkhorn_iters)
        ctx.cost_scale = float(cost_scale)
        ctx.solve_dtype_code = int(solve_dtype_code)
        ctx.cache_dtype_code = int(cache_dtype_code)
        ctx.gmres_iters = int(gmres_iters)
        ctx.gmres_tol = float(gmres_tol)
        ctx.ridge = float(ridge)
        ctx.fused_code = int(fused_code)
        ctx.direct_code = int(direct_code)
        ctx.direct_block_n = int(direct_block_n)
        ctx.direct_block_m = int(direct_block_m)
        ctx.matvec_block_n = int(matvec_block_n)
        ctx.matvec_block_m = int(matvec_block_m)
        ctx.implicit_add_block_m = int(implicit_add_block_m)
        ctx.triton_num_warps = int(triton_num_warps)
        return row_loss.mean()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        residuals, target, u, v, v_direct, direct_lse, direct_row_loss = ctx.saved_tensors
        if int(ctx.direct_code) == 1:
            from .pnw_triton import triton_pnw_row_loss_tiled_backward_from_state

            row_grad = (
                grad_output.detach().to(device=residuals.device, dtype=torch.float32).reshape(())
                / float(residuals.shape[0])
            ).expand((residuals.shape[0],)).contiguous()
            direct_grad, grad_v = triton_pnw_row_loss_tiled_backward_from_state(
                residuals.detach(),
                target.detach(),
                v_direct.detach(),
                direct_lse.detach(),
                direct_row_loss.detach(),
                row_grad,
                epsilon=float(ctx.epsilon),
                cost_scale=float(ctx.cost_scale),
                block_n=int(ctx.direct_block_n),
                block_m=int(ctx.direct_block_m),
                num_warps=int(ctx.triton_num_warps),
            )
        else:
            from .pnw_triton import triton_pnw_row_loss

            with torch.enable_grad():
                residuals_leaf = residuals.detach().clone().requires_grad_(True)
                v_leaf = v_direct.detach().clone().requires_grad_(True)
                row_loss = triton_pnw_row_loss(
                    residuals_leaf,
                    target.detach(),
                    v_leaf,
                    epsilon=float(ctx.epsilon),
                    cost_scale=float(ctx.cost_scale),
                )
                weighted_loss = grad_output.to(dtype=row_loss.dtype) * row_loss.mean()
                direct_grad, grad_v = torch.autograd.grad(weighted_loss, (residuals_leaf, v_leaf))

        solve_dtype = torch.float64 if int(ctx.solve_dtype_code) == 64 else torch.float32
        work_res = residuals.detach().to(dtype=solve_dtype)
        work_target = target.detach().to(dtype=solve_dtype)
        work_u = u.to(dtype=solve_dtype)
        work_v = v.to(dtype=solve_dtype)
        cost = 0.5 * torch.cdist(work_res, work_target).pow(2) / float(ctx.cost_scale)
        cache_dtype = _dtype_from_code(int(ctx.cache_dtype_code))
        a = torch.softmax(work_v[None, :] - cost / float(ctx.epsilon), dim=1).to(dtype=cache_dtype)
        b = torch.softmax(work_u[:, None] - cost / float(ctx.epsilon), dim=0).to(dtype=cache_dtype)
        rhs = _project_zero_mean(grad_v.detach().to(dtype=solve_dtype))
        use_fused_matvec = (
            int(ctx.fused_code) in {2, 3}
            and solve_dtype == torch.float32
            and cache_dtype == torch.float32
            and a.device.type == "cuda"
        )

        def matvec(q: torch.Tensor) -> torch.Tensor:
            if use_fused_matvec:
                from .pnw_triton import triton_cached_ab_matvec

                abq = triton_cached_ab_matvec(
                    a,
                    b,
                    q,
                    block_n=int(ctx.matvec_block_n),
                    block_m=int(ctx.matvec_block_m),
                    num_warps=int(ctx.triton_num_warps),
                )
                out = q - abq.to(dtype=solve_dtype)
            else:
                q_cache = q.to(dtype=cache_dtype)
                bq = (b @ q_cache).to(dtype=solve_dtype)
                out = q - (a.T @ bq.to(dtype=cache_dtype)).to(dtype=solve_dtype)
            if float(ctx.ridge) > 0:
                out = out + float(ctx.ridge) * q
            return _project_zero_mean(out)

        lam_v = _gmres_matrix_free(
            matvec,
            rhs,
            max_iter=int(ctx.gmres_iters),
            tol=float(ctx.gmres_tol),
            matvec_returns_projected=True,
        )
        lam_v_cache = lam_v.to(dtype=cache_dtype)
        lam_u = -(b @ lam_v_cache).to(dtype=solve_dtype)
        use_fused_add = (
            int(ctx.fused_code) % 100 in {1, 3}
            and solve_dtype == torch.float32
            and cache_dtype == torch.float32
            and a.device.type == "cuda"
        )
        use_gemm_add = int(ctx.fused_code) % 100 == 4
        if use_gemm_add:
            implicit_add = _gemm_implicit_add_cached_ab(
                work_res,
                work_target,
                a,
                b,
                lam_u,
                lam_v,
                epsilon=float(ctx.epsilon),
                cost_scale=float(ctx.cost_scale),
            )
        elif use_fused_add:
            from .pnw_triton import triton_cached_ab_implicit_add

            implicit_add = triton_cached_ab_implicit_add(
                work_res,
                work_target,
                a,
                b,
                lam_u,
                lam_v,
                epsilon=float(ctx.epsilon),
                cost_scale=float(ctx.cost_scale),
                block_m=int(ctx.implicit_add_block_m),
                num_warps=int(ctx.triton_num_warps),
            )
        else:
            coeff = lam_u[:, None] * a.to(dtype=solve_dtype) + b.to(dtype=solve_dtype) * lam_v[None, :]
            implicit_add = torch.sum(coeff[:, :, None] * (work_res[:, None, :] - work_target[None, :, :]), dim=1) / (
                float(ctx.epsilon) * float(ctx.cost_scale)
            )
        return (
            direct_grad + implicit_add.to(dtype=direct_grad.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def pnw_scores(
    residuals: torch.Tensor,
    target: torch.Tensor,
    *,
    ot_estimator: str = OT_ESTIMATOR_PNW,
    epsilon: float,
    sinkhorn_iters: int,
    cost_scale: float,
    topk_barycentric: int | None = None,
    chunk_size: int | None = None,
    detach_sinkhorn_gradient: bool = False,
    backend: str = DENSE_TRITON_BACKEND,
    implicit_solve_dtype: str = "float32",
    implicit_gmres_iters: int = 30,
    implicit_gmres_tol: float = 1e-6,
    implicit_ridge: float = 1e-8,
    implicit_chunk_size: int | None = None,
    implicit_cache_dtype: str = "float32",
    sinkhorn_warm_start_id: int = 0,
    triton_direct_block_n: int = 8,
    triton_direct_block_m: int = 1024,
    triton_matvec_block_n: int = 32,
    triton_matvec_block_m: int = 256,
    triton_implicit_add_block_m: int = 256,
    triton_num_warps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the dense PNW full-risk objective, scores, and target dual scaling."""

    _, backend = normalize_ot_estimator_backend(ot_estimator, backend)

    if backend == DENSE_TRITON_BACKEND and (
        residuals.device.type != "cuda" or residuals.dtype != torch.float32 or target.dtype != torch.float32
    ):
        backend = DENSE_PYTORCH_BACKEND

    if backend == DENSE_TRITON_BACKEND:
        if residuals.dtype != torch.float32 or target.dtype != torch.float32:
            raise TypeError("dense_triton PNW currently supports float32 tensors only")
        if topk_barycentric not in (None, 0):
            raise ValueError("dense_triton PNW currently supports full softmax scores, not top-k barycentric scores")
        solve_dtype = str(implicit_solve_dtype).lower()
        if solve_dtype not in {"float32", "float64"}:
            raise ValueError("implicit_solve_dtype must be 'float32' or 'float64'")
        solve_dtype_code = 64 if solve_dtype == "float64" else 32
        cache_dtype_code = _dtype_code_from_name(implicit_cache_dtype)
        fused_code = 4
        direct_code = 1
        loss = _ImplicitCachedABPNWLoss.apply(
            residuals,
            target,
            float(epsilon),
            int(sinkhorn_iters),
            float(cost_scale),
            int(solve_dtype_code),
            int(cache_dtype_code),
            int(implicit_gmres_iters),
            float(implicit_gmres_tol),
            float(implicit_ridge),
            int(fused_code),
            int(direct_code),
            int(sinkhorn_warm_start_id),
            int(triton_direct_block_n),
            int(triton_direct_block_m),
            int(triton_matvec_block_n),
            int(triton_matvec_block_m),
            int(triton_implicit_add_block_m),
            int(triton_num_warps),
        )
        scores = torch.empty((0, residuals.shape[1]), device=residuals.device, dtype=residuals.dtype)
        v = torch.empty((target.shape[0],), device=target.device, dtype=target.dtype)
        return loss, scores, v

    if backend != DENSE_PYTORCH_BACKEND:
        raise ValueError(f"Unknown PNW backend: {backend}")

    if bool(detach_sinkhorn_gradient):
        with torch.no_grad():
            _, v, cost = sinkhorn_log_scalings(
                residuals.detach(),
                target.detach(),
                epsilon=epsilon,
                sinkhorn_iters=sinkhorn_iters,
                cost_scale=cost_scale,
                chunk_size=chunk_size,
            )
        v = v.detach()
        cost = cost.detach()
    else:
        _, v, cost = sinkhorn_log_scalings(
            residuals,
            target,
            epsilon=epsilon,
            sinkhorn_iters=sinkhorn_iters,
            cost_scale=cost_scale,
            chunk_size=chunk_size,
        )
    topk = 0 if topk_barycentric is None else int(topk_barycentric)
    chunk = 0 if chunk_size is None else int(chunk_size)
    if chunk > 0 and chunk < target.shape[0] and topk > 0 and topk < target.shape[0]:
        top_values = None
        top_indices = None
        for start in range(0, target.shape[0], chunk):
            target_chunk = target[start : start + chunk]
            cost_chunk = 0.5 * torch.cdist(residuals, target_chunk).pow(2) / float(cost_scale)
            logits_chunk = v[start : start + target_chunk.shape[0]][None, :] - cost_chunk / float(epsilon)
            k_chunk = min(topk, target_chunk.shape[0])
            values_chunk, local_indices = torch.topk(logits_chunk, k=k_chunk, dim=1)
            indices_chunk = local_indices + int(start)
            if top_values is None:
                top_values = values_chunk
                top_indices = indices_chunk
            else:
                values_all = torch.cat([top_values, values_chunk], dim=1)
                indices_all = torch.cat([top_indices, indices_chunk], dim=1)
                top_values, select = torch.topk(values_all, k=topk, dim=1)
                top_indices = torch.gather(indices_all, dim=1, index=select)
        weights = torch.softmax(top_values, dim=1)
        scores = torch.sum(weights[..., None] * target[top_indices], dim=1)
        loss = torch.mean(torch.sum(residuals * scores, dim=1))
        return loss, scores, v

    if chunk > 0 and chunk < target.shape[0]:
        # Chunked Sinkhorn without top-k score truncation is supported by falling
        # back to the dense score only for current diagnostics. The memory-saving
        # path should be used together with ``topk_barycentric``.
        cost = 0.5 * torch.cdist(residuals, target).pow(2) / float(cost_scale)

    logits = v[None, :] - cost / float(epsilon)
    if topk > 0 and topk < target.shape[0]:
        top_logits, top_idx = torch.topk(logits, k=topk, dim=1)
        weights = torch.softmax(top_logits, dim=1)
        scores = torch.sum(weights[..., None] * target[top_idx], dim=1)
    else:
        weights = torch.softmax(logits, dim=1)
        scores = weights @ target
    loss = torch.mean(torch.sum(residuals * scores, dim=1))
    return loss, scores, v


def sample_pnw_reference(
    *,
    n: int,
    d: int,
    config: PNWEntropicOTConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample the fixed finite reference cloud used by PNW OT-rank."""

    if config.reference_values is not None:
        reference = torch.as_tensor(config.reference_values, device=device, dtype=dtype)
        expected_shape = (int(n), int(d))
        if tuple(reference.shape) != expected_shape:
            raise ValueError(
                f"PNW reference_values must have shape {expected_shape}, got {tuple(reference.shape)}"
            )
        if not bool(torch.isfinite(reference).all()):
            raise ValueError("PNW reference_values must contain only finite values")
        return reference

    gen_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(int(config.seed))
    return sample_reference(
        int(n),
        int(d),
        generator,
        kind=str(config.reference_distribution),
        radius=float(config.reference_radius),
        device=device,
        dtype=dtype,
        seed=int(config.seed),
    )


@dataclass
class PNWEntropicOTMap:
    """Fitted PNW map extension for fixed residual samples.

    This is provided for diagnostics and interface completeness. The main
    full/profile regression estimator recomputes the PNW score inside each
    optimization step instead of freezing this fitted map.
    """

    target: torch.Tensor
    dual_v: torch.Tensor
    epsilon: float
    cost_scale: float

    def transform(self, residuals: torch.Tensor, *, create_graph: bool = False) -> torch.Tensor:
        source = residuals if create_graph else residuals.detach()
        cost = 0.5 * torch.cdist(source, self.target).pow(2) / float(self.cost_scale)
        logits = self.dual_v[None, :] - cost / float(self.epsilon)
        return torch.softmax(logits, dim=1) @ self.target

    def potential(self, residuals: torch.Tensor, *, create_graph: bool = False) -> torch.Tensor:
        source = residuals if create_graph else residuals.detach()
        cost = 0.5 * torch.cdist(source, self.target).pow(2) / float(self.cost_scale)
        logits = self.dual_v[None, :] - cost / float(self.epsilon)
        return float(self.epsilon) * torch.logsumexp(logits, dim=1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.detach().cpu(),
            "dual_v": self.dual_v.detach().cpu(),
            "epsilon": float(self.epsilon),
            "cost_scale": float(self.cost_scale),
        }


@dataclass
class PNWEntropicOTMapEstimator:
    """Fit a direct entropic OT map from residuals to a uniform-ball reference."""

    config: PNWEntropicOTConfig = field(default_factory=PNWEntropicOTConfig)

    def fit(
        self,
        residuals: torch.Tensor,
        init_state: dict[str, Any] | None = None,
        save_dir: str | Path | None = None,
    ) -> FittedOTMap:
        _ = init_state
        target = sample_pnw_reference(
            n=int(self.config.m_ref),
            d=residuals.shape[1],
            config=self.config,
            device=residuals.device,
            dtype=residuals.dtype,
        )
        cost_scale = (
            float(self.config.cost_scale)
            if self.config.cost_scale is not None
            else median_cost_scale(residuals, target)
        )
        _, _, dual_v = pnw_scores(
            residuals,
            target,
            epsilon=float(self.config.epsilon),
            sinkhorn_iters=int(self.config.sinkhorn_iters),
            cost_scale=cost_scale,
            topk_barycentric=self.config.topk_barycentric,
            chunk_size=self.config.chunk_size,
            detach_sinkhorn_gradient=self.config.detach_sinkhorn_gradient,
            backend=DENSE_PYTORCH_BACKEND,
        )
        fitted = PNWEntropicOTMap(
            target=target.detach(),
            dual_v=dual_v.detach(),
            epsilon=float(self.config.epsilon),
            cost_scale=cost_scale,
        )
        if save_dir is not None:
            path = Path(save_dir)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(fitted.state_dict(), path / "pnw_ot_map.pt")
        return fitted
