"""Dense Triton kernels for PNW pairwise reductions.

This module contains the GPU kernels used by the dense PNW backend.  Sparse
and other exploratory kernels are archived outside the active package tree.
"""

from __future__ import annotations

import math
import torch

try:  # pragma: no cover - Triton is only available on the GPU server.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


def _triton_jit(function):
    """Apply ``triton.jit`` when available without breaking CPU imports."""

    if triton is None:
        return function
    return triton.jit(function)


def _require_triton() -> None:
    if triton is None or tl is None:
        raise ImportError("Triton is required for otrank.ot_maps.pnw_triton")
    if not torch.cuda.is_available():
        raise RuntimeError("Triton PNW kernels require a CUDA device")


@_triton_jit
def _row_logsumexp_forward_kernel(
    x_ptr,
    z_ptr,
    v_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    m_ref: tl.constexpr,
    d_dim: tl.constexpr,
    eps_scale: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_xd: tl.constexpr,
    stride_zm: tl.constexpr,
    stride_zd: tl.constexpr,
    block_m: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_m)
    x_base = x_ptr + row * stride_xn
    # Triton 2.1 forbids zero-dimensional block constructors. Plain scalar
    # initializers are promoted to the reduction result type inside the loop.
    max_val = -float("inf")
    sum_exp = 0.0

    for start in range(0, m_ref, block_m):
        cols = start + offsets
        mask = cols < m_ref
        dist = tl.zeros((block_m,), dtype=tl.float32)
        for d in range(0, d_dim):
            x_val = tl.load(x_base + d * stride_xd).to(tl.float32)
            z_val = tl.load(z_ptr + cols * stride_zm + d * stride_zd, mask=mask, other=0.0).to(tl.float32)
            diff = x_val - z_val
            dist += diff * diff
        v_val = tl.load(v_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        logits = v_val - 0.5 * dist / eps_scale
        logits = tl.where(mask, logits, -float("inf"))
        block_max = tl.max(logits, axis=0)
        block_sum = tl.sum(tl.exp(logits - block_max), axis=0)
        if start == 0:
            max_val = block_max
            sum_exp = block_sum
        else:
            new_max = tl.maximum(max_val, block_max)
            sum_exp = sum_exp * tl.exp(max_val - new_max) + block_sum * tl.exp(block_max - new_max)
            max_val = new_max

    tl.store(out_ptr + row, max_val + tl.log(sum_exp))


@_triton_jit
def _row_logsumexp_backward_kernel(
    x_ptr,
    z_ptr,
    v_ptr,
    out_ptr,
    grad_out_ptr,
    grad_x_ptr,
    grad_z_ptr,
    grad_v_ptr,
    n_rows: tl.constexpr,
    m_ref: tl.constexpr,
    d_dim: tl.constexpr,
    eps_scale: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_xd: tl.constexpr,
    stride_zm: tl.constexpr,
    stride_zd: tl.constexpr,
    stride_gxn: tl.constexpr,
    stride_gxd: tl.constexpr,
    stride_gzm: tl.constexpr,
    stride_gzd: tl.constexpr,
    needs_grad_x: tl.constexpr,
    needs_grad_z: tl.constexpr,
    block_m: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_m)
    x_base = x_ptr + row * stride_xn
    grad_out = tl.load(grad_out_ptr + row).to(tl.float32)
    lse = tl.load(out_ptr + row).to(tl.float32)

    for start in range(0, m_ref, block_m):
        cols = start + offsets
        mask = cols < m_ref
        dist = tl.zeros((block_m,), dtype=tl.float32)
        for d in range(0, d_dim):
            x_val = tl.load(x_base + d * stride_xd).to(tl.float32)
            z_val = tl.load(z_ptr + cols * stride_zm + d * stride_zd, mask=mask, other=0.0).to(tl.float32)
            diff = x_val - z_val
            dist += diff * diff
        v_val = tl.load(v_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        logits = v_val - 0.5 * dist / eps_scale
        weights = tl.exp(logits - lse)
        weights = tl.where(mask, weights, 0.0)
        grad_v = grad_out * weights
        tl.atomic_add(grad_v_ptr + cols, grad_v, sem="relaxed", mask=mask)

        # d logit_ij / d x_i = (z_j - x_i) / eps_scale.
        for d in range(0, d_dim):
            x_val = tl.load(x_base + d * stride_xd).to(tl.float32)
            z_val = tl.load(z_ptr + cols * stride_zm + d * stride_zd, mask=mask, other=0.0).to(tl.float32)
            pair_grad = grad_out * weights * (z_val - x_val) / eps_scale
            if needs_grad_x:
                grad_component = tl.sum(pair_grad, axis=0)
                tl.atomic_add(grad_x_ptr + row * stride_gxn + d * stride_gxd, grad_component, sem="relaxed")
            if needs_grad_z:
                tl.atomic_add(grad_z_ptr + cols * stride_gzm + d * stride_gzd, -pair_grad, sem="relaxed", mask=mask)


class _TritonRowLogsumexp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, z: torch.Tensor, v: torch.Tensor, eps_scale: float, block_m: int) -> torch.Tensor:
        _require_triton()
        if x.device.type != "cuda" or z.device.type != "cuda" or v.device.type != "cuda":
            raise RuntimeError("Triton row_logsumexp inputs must be CUDA tensors")
        if x.dtype != torch.float32 or z.dtype != torch.float32 or v.dtype != torch.float32:
            raise TypeError("Triton row_logsumexp currently supports float32 tensors only")
        x = x.contiguous()
        z = z.contiguous()
        v = v.contiguous()
        n_rows, d_dim = x.shape
        m_ref = z.shape[0]
        if z.shape[1] != d_dim:
            raise ValueError("x and z must have the same feature dimension")
        out = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
        block = int(block_m)
        _row_logsumexp_forward_kernel[(n_rows,)](
            x,
            z,
            v,
            out,
            n_rows,
            m_ref,
            d_dim,
            float(eps_scale),
            x.stride(0),
            x.stride(1),
            z.stride(0),
            z.stride(1),
            block_m=block,
            num_warps=8,
        )
        ctx.save_for_backward(x, z, v, out)
        ctx.eps_scale = float(eps_scale)
        ctx.block_m = block
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, z, v, out = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        needs_x, needs_z, needs_v = ctx.needs_input_grad[:3]
        grad_x = torch.zeros_like(x) if needs_x else torch.empty((1,), device=x.device, dtype=x.dtype)
        grad_z = torch.zeros_like(z) if needs_z else torch.empty((1,), device=z.device, dtype=z.dtype)
        grad_v = torch.zeros_like(v)
        n_rows, d_dim = x.shape
        m_ref = z.shape[0]
        _row_logsumexp_backward_kernel[(n_rows,)](
            x,
            z,
            v,
            out,
            grad_out,
            grad_x,
            grad_z,
            grad_v,
            n_rows,
            m_ref,
            d_dim,
            float(ctx.eps_scale),
            x.stride(0),
            x.stride(1),
            z.stride(0),
            z.stride(1),
            grad_x.stride(0) if needs_x else 0,
            grad_x.stride(1) if needs_x else 0,
            grad_z.stride(0) if needs_z else 0,
            grad_z.stride(1) if needs_z else 0,
            needs_grad_x=bool(needs_x),
            needs_grad_z=bool(needs_z),
            block_m=int(ctx.block_m),
            num_warps=8,
        )
        return (grad_x if needs_x else None), (grad_z if needs_z else None), (grad_v if needs_v else None), None, None


def triton_row_logsumexp(
    x: torch.Tensor,
    z: torch.Tensor,
    v: torch.Tensor,
    *,
    epsilon: float,
    cost_scale: float,
    block_m: int = 1024,
) -> torch.Tensor:
    """Compute ``logsumexp_j(v_j - ||x_i-z_j||^2/(2 eps scale))``.

    This is the first custom-backward Triton building block for PNW.  It
    returns gradients with respect to ``x`` and ``v`` and treats the reference
    cloud ``z`` as fixed.
    """

    eps_scale = float(epsilon) * float(cost_scale)
    if eps_scale <= 0 or not math.isfinite(eps_scale):
        raise ValueError("epsilon * cost_scale must be positive and finite")
    return _TritonRowLogsumexp.apply(x, z, v, eps_scale, int(block_m))


@_triton_jit
def _pnw_row_loss_forward_kernel(
    x_ptr,
    z_ptr,
    v_ptr,
    lse_ptr,
    loss_ptr,
    n_rows: tl.constexpr,
    m_ref: tl.constexpr,
    d_dim: tl.constexpr,
    eps_scale: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_xd: tl.constexpr,
    stride_zm: tl.constexpr,
    stride_zd: tl.constexpr,
    block_m: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_m)
    x_base = x_ptr + row * stride_xn
    max_val = -float("inf")
    sum_exp = 0.0

    for start in range(0, m_ref, block_m):
        cols = start + offsets
        mask = cols < m_ref
        dist = tl.zeros((block_m,), dtype=tl.float32)
        for d in range(0, d_dim):
            x_val = tl.load(x_base + d * stride_xd).to(tl.float32)
            z_val = tl.load(z_ptr + cols * stride_zm + d * stride_zd, mask=mask, other=0.0).to(tl.float32)
            diff = x_val - z_val
            dist += diff * diff
        v_val = tl.load(v_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        logits = v_val - 0.5 * dist / eps_scale
        logits = tl.where(mask, logits, -float("inf"))
        block_max = tl.max(logits, axis=0)
        block_sum = tl.sum(tl.exp(logits - block_max), axis=0)
        if start == 0:
            max_val = block_max
            sum_exp = block_sum
        else:
            new_max = tl.maximum(max_val, block_max)
            sum_exp = sum_exp * tl.exp(max_val - new_max) + block_sum * tl.exp(block_max - new_max)
            max_val = new_max

    lse = max_val + tl.log(sum_exp)
    tl.store(lse_ptr + row, lse)

    row_loss = 0.0
    for start in range(0, m_ref, block_m):
        cols = start + offsets
        mask = cols < m_ref
        dist = tl.zeros((block_m,), dtype=tl.float32)
        dot_xz = tl.zeros((block_m,), dtype=tl.float32)
        for d in range(0, d_dim):
            x_val = tl.load(x_base + d * stride_xd).to(tl.float32)
            z_val = tl.load(z_ptr + cols * stride_zm + d * stride_zd, mask=mask, other=0.0).to(tl.float32)
            diff = x_val - z_val
            dist += diff * diff
            dot_xz += x_val * z_val
        v_val = tl.load(v_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        logits = v_val - 0.5 * dist / eps_scale
        weights = tl.exp(logits - lse)
        weights = tl.where(mask, weights, 0.0)
        row_loss += tl.sum(weights * dot_xz, axis=0)
    tl.store(loss_ptr + row, row_loss)


@_triton_jit
def _pnw_row_loss_backward_kernel(
    x_ptr,
    z_ptr,
    v_ptr,
    lse_ptr,
    loss_ptr,
    grad_out_ptr,
    grad_x_ptr,
    grad_v_ptr,
    n_rows: tl.constexpr,
    m_ref: tl.constexpr,
    d_dim: tl.constexpr,
    eps_scale: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_xd: tl.constexpr,
    stride_zm: tl.constexpr,
    stride_zd: tl.constexpr,
    stride_gxn: tl.constexpr,
    stride_gxd: tl.constexpr,
    block_m: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_m)
    x_base = x_ptr + row * stride_xn
    lse = tl.load(lse_ptr + row).to(tl.float32)
    row_loss = tl.load(loss_ptr + row).to(tl.float32)
    grad_out = tl.load(grad_out_ptr + row).to(tl.float32)

    for start in range(0, m_ref, block_m):
        cols = start + offsets
        mask = cols < m_ref
        dist = tl.zeros((block_m,), dtype=tl.float32)
        dot_xz = tl.zeros((block_m,), dtype=tl.float32)
        for d in range(0, d_dim):
            x_val = tl.load(x_base + d * stride_xd).to(tl.float32)
            z_val = tl.load(z_ptr + cols * stride_zm + d * stride_zd, mask=mask, other=0.0).to(tl.float32)
            diff = x_val - z_val
            dist += diff * diff
            dot_xz += x_val * z_val

        v_val = tl.load(v_ptr + cols, mask=mask, other=-float("inf")).to(tl.float32)
        logits = v_val - 0.5 * dist / eps_scale
        weights = tl.exp(logits - lse)
        weights = tl.where(mask, weights, 0.0)

        grad_v = grad_out * weights * (dot_xz - row_loss)
        tl.atomic_add(grad_v_ptr + cols, grad_v, sem="relaxed", mask=mask)

        for d in range(0, d_dim):
            z_val = tl.load(z_ptr + cols * stride_zm + d * stride_zd, mask=mask, other=0.0).to(tl.float32)
            s_chunk = tl.sum(weights * z_val, axis=0)
            t_chunk = tl.sum(weights * dot_xz * z_val, axis=0)
            grad_x = grad_out * (s_chunk + (t_chunk - row_loss * s_chunk) / eps_scale)
            tl.atomic_add(grad_x_ptr + row * stride_gxn + d * stride_gxd, grad_x, sem="relaxed")


class _TritonPNWRowLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, z: torch.Tensor, v: torch.Tensor, eps_scale: float, block_m: int) -> torch.Tensor:
        _require_triton()
        if x.device.type != "cuda" or z.device.type != "cuda" or v.device.type != "cuda":
            raise RuntimeError("Triton PNW row loss inputs must be CUDA tensors")
        if x.dtype != torch.float32 or z.dtype != torch.float32 or v.dtype != torch.float32:
            raise TypeError("Triton PNW row loss currently supports float32 tensors only")
        x = x.contiguous()
        z = z.contiguous()
        v = v.contiguous()
        n_rows, d_dim = x.shape
        m_ref = z.shape[0]
        if z.shape[1] != d_dim:
            raise ValueError("x and z must have the same feature dimension")
        lse = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
        row_loss = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
        _pnw_row_loss_forward_kernel[(n_rows,)](
            x,
            z,
            v,
            lse,
            row_loss,
            n_rows,
            m_ref,
            d_dim,
            float(eps_scale),
            x.stride(0),
            x.stride(1),
            z.stride(0),
            z.stride(1),
            block_m=int(block_m),
            num_warps=8,
        )
        ctx.save_for_backward(x, z, v, lse, row_loss)
        ctx.eps_scale = float(eps_scale)
        ctx.block_m = int(block_m)
        return row_loss

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, z, v, lse, row_loss = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        needs_x, needs_z, needs_v = ctx.needs_input_grad[:3]
        if needs_z:
            raise RuntimeError("Triton PNW row loss treats reference points as fixed")
        grad_x = torch.zeros_like(x) if needs_x else torch.empty((1,), device=x.device, dtype=x.dtype)
        grad_v = torch.zeros_like(v) if needs_v else torch.empty((1,), device=v.device, dtype=v.dtype)
        n_rows, d_dim = x.shape
        m_ref = z.shape[0]
        _pnw_row_loss_backward_kernel[(n_rows,)](
            x,
            z,
            v,
            lse,
            row_loss,
            grad_out,
            grad_x,
            grad_v,
            n_rows,
            m_ref,
            d_dim,
            float(ctx.eps_scale),
            x.stride(0),
            x.stride(1),
            z.stride(0),
            z.stride(1),
            grad_x.stride(0) if needs_x else 0,
            grad_x.stride(1) if needs_x else 0,
            block_m=int(ctx.block_m),
            num_warps=8,
        )
        return (grad_x if needs_x else None), None, (grad_v if needs_v else None), None, None


def triton_pnw_row_loss(
    x: torch.Tensor,
    z: torch.Tensor,
    v: torch.Tensor,
    *,
    epsilon: float,
    cost_scale: float,
    block_m: int = 1024,
) -> torch.Tensor:
    """Return per-row values ``x_i^T sum_j softmax(logit_ij) z_j``.

    The backward pass returns gradients with respect to ``x`` and ``v`` and
    treats ``z`` as a fixed reference cloud.
    """

    eps_scale = float(epsilon) * float(cost_scale)
    if eps_scale <= 0 or not math.isfinite(eps_scale):
        raise ValueError("epsilon * cost_scale must be positive and finite")
    return _TritonPNWRowLoss.apply(x, z, v, eps_scale, int(block_m))


@_triton_jit
def _pnw_row_loss_tiled_forward_kernel(
    x_ptr,
    z_ptr,
    v_ptr,
    lse_ptr,
    loss_ptr,
    n_rows: tl.constexpr,
    m_ref: tl.constexpr,
    d_dim: tl.constexpr,
    eps_scale: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_xd: tl.constexpr,
    stride_zm: tl.constexpr,
    stride_zd: tl.constexpr,
    block_n: tl.constexpr,
    block_m: tl.constexpr,
):
    block_id = tl.program_id(0)
    rows = block_id * block_n + tl.arange(0, block_n)
    cols_offsets = tl.arange(0, block_m)
    row_mask = rows < n_rows

    max_val = tl.full((block_n,), -float("inf"), tl.float32)
    sum_exp = tl.full((block_n,), 0.0, tl.float32)

    for start in range(0, m_ref, block_m):
        cols = start + cols_offsets
        col_mask = cols < m_ref
        dist = tl.zeros((block_n, block_m), dtype=tl.float32)
        for d in range(0, d_dim):
            x_val = tl.load(x_ptr + rows[:, None] * stride_xn + d * stride_xd, mask=row_mask[:, None], other=0.0).to(tl.float32)
            z_val = tl.load(z_ptr + cols[None, :] * stride_zm + d * stride_zd, mask=col_mask[None, :], other=0.0).to(tl.float32)
            diff = x_val - z_val
            dist += diff * diff
        v_val = tl.load(v_ptr + cols, mask=col_mask, other=-float("inf")).to(tl.float32)
        logits = v_val[None, :] - 0.5 * dist / eps_scale
        logits = tl.where(row_mask[:, None] & col_mask[None, :], logits, -float("inf"))
        block_max = tl.max(logits, axis=1)
        block_sum = tl.sum(tl.exp(logits - block_max[:, None]), axis=1)
        if start == 0:
            max_val = block_max
            sum_exp = block_sum
        else:
            new_max = tl.maximum(max_val, block_max)
            sum_exp = sum_exp * tl.exp(max_val - new_max) + block_sum * tl.exp(block_max - new_max)
            max_val = new_max

    lse = max_val + tl.log(sum_exp)
    tl.store(lse_ptr + rows, lse, mask=row_mask)

    row_loss = tl.zeros((block_n,), dtype=tl.float32)
    for start in range(0, m_ref, block_m):
        cols = start + cols_offsets
        col_mask = cols < m_ref
        dist = tl.zeros((block_n, block_m), dtype=tl.float32)
        dot_xz = tl.zeros((block_n, block_m), dtype=tl.float32)
        for d in range(0, d_dim):
            x_val = tl.load(x_ptr + rows[:, None] * stride_xn + d * stride_xd, mask=row_mask[:, None], other=0.0).to(tl.float32)
            z_val = tl.load(z_ptr + cols[None, :] * stride_zm + d * stride_zd, mask=col_mask[None, :], other=0.0).to(tl.float32)
            diff = x_val - z_val
            dist += diff * diff
            dot_xz += x_val * z_val
        v_val = tl.load(v_ptr + cols, mask=col_mask, other=-float("inf")).to(tl.float32)
        logits = v_val[None, :] - 0.5 * dist / eps_scale
        weights = tl.exp(logits - lse[:, None])
        weights = tl.where(row_mask[:, None] & col_mask[None, :], weights, 0.0)
        row_loss += tl.sum(weights * dot_xz, axis=1)
    tl.store(loss_ptr + rows, row_loss, mask=row_mask)


@_triton_jit
def _pnw_row_loss_tiled_backward_kernel(
    x_ptr,
    z_ptr,
    v_ptr,
    lse_ptr,
    loss_ptr,
    grad_out_ptr,
    grad_x_ptr,
    partial_grad_v_ptr,
    n_rows: tl.constexpr,
    m_ref: tl.constexpr,
    d_dim: tl.constexpr,
    eps_scale: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_xd: tl.constexpr,
    stride_zm: tl.constexpr,
    stride_zd: tl.constexpr,
    stride_gxn: tl.constexpr,
    stride_gxd: tl.constexpr,
    num_row_blocks: tl.constexpr,
    block_n: tl.constexpr,
    block_m: tl.constexpr,
):
    block_id = tl.program_id(0)
    rows = block_id * block_n + tl.arange(0, block_n)
    cols_offsets = tl.arange(0, block_m)
    row_mask = rows < n_rows
    lse = tl.load(lse_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    row_loss = tl.load(loss_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    grad_out = tl.load(grad_out_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

    for start in range(0, m_ref, block_m):
        cols = start + cols_offsets
        col_mask = cols < m_ref
        dist = tl.zeros((block_n, block_m), dtype=tl.float32)
        dot_xz = tl.zeros((block_n, block_m), dtype=tl.float32)
        for d in range(0, d_dim):
            x_val = tl.load(x_ptr + rows[:, None] * stride_xn + d * stride_xd, mask=row_mask[:, None], other=0.0).to(tl.float32)
            z_val = tl.load(z_ptr + cols[None, :] * stride_zm + d * stride_zd, mask=col_mask[None, :], other=0.0).to(tl.float32)
            diff = x_val - z_val
            dist += diff * diff
            dot_xz += x_val * z_val

        v_val = tl.load(v_ptr + cols, mask=col_mask, other=-float("inf")).to(tl.float32)
        logits = v_val[None, :] - 0.5 * dist / eps_scale
        weights = tl.exp(logits - lse[:, None])
        weights = tl.where(row_mask[:, None] & col_mask[None, :], weights, 0.0)
        go_w = grad_out[:, None] * weights
        partial_grad_v = tl.sum(go_w * (dot_xz - row_loss[:, None]), axis=0)
        tl.store(partial_grad_v_ptr + block_id * m_ref + cols, partial_grad_v, mask=col_mask)

        for d in range(0, d_dim):
            z_val = tl.load(z_ptr + cols[None, :] * stride_zm + d * stride_zd, mask=col_mask[None, :], other=0.0).to(tl.float32)
            s_chunk = tl.sum(weights * z_val, axis=1)
            t_chunk = tl.sum(weights * dot_xz * z_val, axis=1)
            grad_x = grad_out * (s_chunk + (t_chunk - row_loss * s_chunk) / eps_scale)
            tl.atomic_add(grad_x_ptr + rows * stride_gxn + d * stride_gxd, grad_x, sem="relaxed", mask=row_mask)


@_triton_jit
def _reduce_partial_grad_v_kernel(
    partial_ptr,
    grad_v_ptr,
    m_ref: tl.constexpr,
    num_row_blocks: tl.constexpr,
    block_m: tl.constexpr,
    block_b: tl.constexpr,
):
    ref_block = tl.program_id(0)
    cols = ref_block * block_m + tl.arange(0, block_m)
    block_offsets = tl.arange(0, block_b)
    col_mask = cols < m_ref
    acc = tl.zeros((block_m,), dtype=tl.float32)
    for start in range(0, num_row_blocks, block_b):
        bids = start + block_offsets
        block_mask = bids < num_row_blocks
        vals = tl.load(partial_ptr + bids[:, None] * m_ref + cols[None, :], mask=block_mask[:, None] & col_mask[None, :], other=0.0)
        acc += tl.sum(vals, axis=0)
    tl.store(grad_v_ptr + cols, acc, mask=col_mask)


class _TritonPNWRowLossTiled(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        z: torch.Tensor,
        v: torch.Tensor,
        eps_scale: float,
        block_n: int,
        block_m: int,
    ) -> torch.Tensor:
        _require_triton()
        if x.device.type != "cuda" or z.device.type != "cuda" or v.device.type != "cuda":
            raise RuntimeError("Triton tiled PNW row loss inputs must be CUDA tensors")
        if x.dtype != torch.float32 or z.dtype != torch.float32 or v.dtype != torch.float32:
            raise TypeError("Triton tiled PNW row loss currently supports float32 tensors only")
        x = x.contiguous()
        z = z.contiguous()
        v = v.contiguous()
        n_rows, d_dim = x.shape
        m_ref = z.shape[0]
        if z.shape[1] != d_dim:
            raise ValueError("x and z must have the same feature dimension")
        row_loss = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
        lse = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
        bn = int(block_n)
        bm = int(block_m)
        num_row_blocks = triton.cdiv(n_rows, bn)
        _pnw_row_loss_tiled_forward_kernel[(num_row_blocks,)](
            x,
            z,
            v,
            lse,
            row_loss,
            n_rows,
            m_ref,
            d_dim,
            float(eps_scale),
            x.stride(0),
            x.stride(1),
            z.stride(0),
            z.stride(1),
            block_n=bn,
            block_m=bm,
            num_warps=8,
        )
        ctx.save_for_backward(x, z, v, lse, row_loss)
        ctx.eps_scale = float(eps_scale)
        ctx.block_n = bn
        ctx.block_m = bm
        ctx.num_row_blocks = int(num_row_blocks)
        return row_loss

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, z, v, lse, row_loss = ctx.saved_tensors
        needs_x, needs_z, needs_v = ctx.needs_input_grad[:3]
        if needs_z:
            raise RuntimeError("Triton tiled PNW row loss treats reference points as fixed")
        grad_out = grad_out.contiguous()
        grad_x = torch.zeros_like(x) if needs_x else torch.empty((1,), device=x.device, dtype=x.dtype)
        partial_grad_v = torch.empty((ctx.num_row_blocks, z.shape[0]), device=x.device, dtype=torch.float32)
        n_rows, d_dim = x.shape
        m_ref = z.shape[0]
        _pnw_row_loss_tiled_backward_kernel[(ctx.num_row_blocks,)](
            x,
            z,
            v,
            lse,
            row_loss,
            grad_out,
            grad_x,
            partial_grad_v,
            n_rows,
            m_ref,
            d_dim,
            float(ctx.eps_scale),
            x.stride(0),
            x.stride(1),
            z.stride(0),
            z.stride(1),
            grad_x.stride(0) if needs_x else 0,
            grad_x.stride(1) if needs_x else 0,
            ctx.num_row_blocks,
            block_n=int(ctx.block_n),
            block_m=int(ctx.block_m),
            num_warps=8,
        )
        grad_v = torch.empty_like(v) if needs_v else torch.empty((1,), device=v.device, dtype=v.dtype)
        _reduce_partial_grad_v_kernel[(triton.cdiv(m_ref, int(ctx.block_m)),)](
            partial_grad_v,
            grad_v,
            m_ref,
            ctx.num_row_blocks,
            block_m=int(ctx.block_m),
            block_b=64,
            num_warps=8,
        )
        return (grad_x if needs_x else None), None, (grad_v if needs_v else None), None, None, None


def triton_pnw_row_loss_tiled(
    x: torch.Tensor,
    z: torch.Tensor,
    v: torch.Tensor,
    *,
    epsilon: float,
    cost_scale: float,
    block_n: int = 16,
    block_m: int = 1024,
) -> torch.Tensor:
    """Tiled low-memory PNW row loss with block-level ``grad_v`` reduction."""

    eps_scale = float(epsilon) * float(cost_scale)
    if eps_scale <= 0 or not math.isfinite(eps_scale):
        raise ValueError("epsilon * cost_scale must be positive and finite")
    return _TritonPNWRowLossTiled.apply(x, z, v, eps_scale, int(block_n), int(block_m))


def triton_pnw_row_loss_tiled_state(
    x: torch.Tensor,
    z: torch.Tensor,
    v: torch.Tensor,
    *,
    epsilon: float,
    cost_scale: float,
    block_n: int = 16,
    block_m: int = 1024,
    num_warps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tiled PNW row loss and row log-normalizers without autograd.

    This exposes the forward state needed by the outer implicit PNW backward so
    it does not need to rerun the direct-loss forward only to recover
    ``grad_x`` and ``grad_v``.
    """

    _require_triton()
    if x.device.type != "cuda" or z.device.type != "cuda" or v.device.type != "cuda":
        raise RuntimeError("Triton tiled PNW row loss inputs must be CUDA tensors")
    if x.dtype != torch.float32 or z.dtype != torch.float32 or v.dtype != torch.float32:
        raise TypeError("Triton tiled PNW row loss currently supports float32 tensors only")
    eps_scale = float(epsilon) * float(cost_scale)
    if eps_scale <= 0 or not math.isfinite(eps_scale):
        raise ValueError("epsilon * cost_scale must be positive and finite")
    x = x.contiguous()
    z = z.contiguous()
    v = v.contiguous()
    n_rows, d_dim = x.shape
    m_ref = z.shape[0]
    if z.shape[1] != d_dim:
        raise ValueError("x and z must have the same feature dimension")
    row_loss = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    lse = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    bn = int(block_n)
    bm = int(block_m)
    num_row_blocks = triton.cdiv(n_rows, bn)
    _pnw_row_loss_tiled_forward_kernel[(num_row_blocks,)](
        x,
        z,
        v,
        lse,
        row_loss,
        n_rows,
        m_ref,
        d_dim,
        float(eps_scale),
        x.stride(0),
        x.stride(1),
        z.stride(0),
        z.stride(1),
        block_n=bn,
        block_m=bm,
        num_warps=int(num_warps),
    )
    return row_loss, lse


def triton_pnw_row_loss_tiled_backward_from_state(
    x: torch.Tensor,
    z: torch.Tensor,
    v: torch.Tensor,
    lse: torch.Tensor,
    row_loss: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    epsilon: float,
    cost_scale: float,
    block_n: int = 16,
    block_m: int = 1024,
    num_warps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct tiled PNW row-loss backward from saved forward state."""

    _require_triton()
    tensors = (x, z, v, lse, row_loss, grad_out)
    if any(t.device.type != "cuda" for t in tensors):
        raise RuntimeError("Triton tiled PNW row loss inputs must be CUDA tensors")
    if any(t.dtype != torch.float32 for t in tensors):
        raise TypeError("Triton tiled PNW row loss currently supports float32 tensors only")
    eps_scale = float(epsilon) * float(cost_scale)
    if eps_scale <= 0 or not math.isfinite(eps_scale):
        raise ValueError("epsilon * cost_scale must be positive and finite")
    x = x.contiguous()
    z = z.contiguous()
    v = v.contiguous()
    lse = lse.contiguous()
    row_loss = row_loss.contiguous()
    grad_out = grad_out.contiguous()
    n_rows, d_dim = x.shape
    m_ref = z.shape[0]
    if z.shape[1] != d_dim:
        raise ValueError("x and z must have the same feature dimension")
    bn = int(block_n)
    bm = int(block_m)
    num_row_blocks = triton.cdiv(n_rows, bn)
    grad_x = torch.zeros_like(x)
    partial_grad_v = torch.empty((num_row_blocks, m_ref), device=x.device, dtype=torch.float32)
    _pnw_row_loss_tiled_backward_kernel[(num_row_blocks,)](
        x,
        z,
        v,
        lse,
        row_loss,
        grad_out,
        grad_x,
        partial_grad_v,
        n_rows,
        m_ref,
        d_dim,
        float(eps_scale),
        x.stride(0),
        x.stride(1),
        z.stride(0),
        z.stride(1),
        grad_x.stride(0),
        grad_x.stride(1),
        num_row_blocks,
        block_n=bn,
        block_m=bm,
        num_warps=int(num_warps),
    )
    grad_v = torch.empty_like(v)
    _reduce_partial_grad_v_kernel[(triton.cdiv(m_ref, bm),)](
        partial_grad_v,
        grad_v,
        m_ref,
        num_row_blocks,
        block_m=bm,
        block_b=64,
        num_warps=int(num_warps),
    )
    return grad_x, grad_v


@_triton_jit
def _cached_ab_row_matvec_kernel(
    mat_ptr,
    q_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    m_ref: tl.constexpr,
    stride_mn: tl.constexpr,
    stride_mm: tl.constexpr,
    block_m: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_m)
    acc = 0.0
    for start in range(0, m_ref, block_m):
        cols = start + offsets
        mask = cols < m_ref
        vals = tl.load(mat_ptr + row * stride_mn + cols * stride_mm, mask=mask, other=0.0).to(tl.float32)
        q = tl.load(q_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals * q, axis=0)
    tl.store(out_ptr + row, acc)


@_triton_jit
def _cached_ab_col_matvec_kernel(
    mat_ptr,
    q_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    m_ref: tl.constexpr,
    stride_mn: tl.constexpr,
    stride_mm: tl.constexpr,
    block_n: tl.constexpr,
    block_m: tl.constexpr,
):
    block = tl.program_id(0)
    cols = block * block_m + tl.arange(0, block_m)
    rows_offsets = tl.arange(0, block_n)
    col_mask = cols < m_ref
    acc = tl.zeros((block_m,), dtype=tl.float32)
    for start in range(0, n_rows, block_n):
        rows = start + rows_offsets
        row_mask = rows < n_rows
        vals = tl.load(
            mat_ptr + rows[:, None] * stride_mn + cols[None, :] * stride_mm,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        q = tl.load(q_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals * q[:, None], axis=0)
    tl.store(out_ptr + cols, acc, mask=col_mask)


def triton_cached_ab_matvec(
    a: torch.Tensor,
    b: torch.Tensor,
    q: torch.Tensor,
    *,
    block_n: int = 32,
    block_m: int = 256,
    num_warps: int = 8,
) -> torch.Tensor:
    """Compute ``A.T @ (B @ q)`` for cached softmax matrices.

    This is an experimental replacement for two PyTorch GEMV calls in the
    implicit PNW backward path. Inputs must be CUDA float32 tensors.
    """

    _require_triton()
    if a.device.type != "cuda" or b.device.type != "cuda" or q.device.type != "cuda":
        raise RuntimeError("triton_cached_ab_matvec inputs must be CUDA tensors")
    if a.dtype != torch.float32 or b.dtype != torch.float32 or q.dtype != torch.float32:
        raise TypeError("triton_cached_ab_matvec currently supports float32 tensors only")
    a = a.contiguous()
    b = b.contiguous()
    q = q.contiguous()
    n_rows, m_ref = a.shape
    if b.shape != a.shape or q.shape != (m_ref,):
        raise ValueError("Expected a and b with shape (n, m), and q with shape (m,)")
    tmp = torch.empty((n_rows,), device=a.device, dtype=torch.float32)
    out = torch.empty((m_ref,), device=a.device, dtype=torch.float32)
    _cached_ab_row_matvec_kernel[(n_rows,)](
        b,
        q,
        tmp,
        n_rows,
        m_ref,
        b.stride(0),
        b.stride(1),
        block_m=int(block_m),
        num_warps=int(num_warps),
    )
    _cached_ab_col_matvec_kernel[(triton.cdiv(m_ref, int(block_m)),)](
        a,
        tmp,
        out,
        n_rows,
        m_ref,
        a.stride(0),
        a.stride(1),
        block_n=int(block_n),
        block_m=int(block_m),
        num_warps=int(num_warps),
    )
    return out


@_triton_jit
def _cached_ab_implicit_add_kernel(
    res_ptr,
    target_ptr,
    a_ptr,
    b_ptr,
    lam_u_ptr,
    lam_v_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    m_ref: tl.constexpr,
    d_dim: tl.constexpr,
    eps_scale: tl.constexpr,
    stride_rn: tl.constexpr,
    stride_rd: tl.constexpr,
    stride_tm: tl.constexpr,
    stride_td: tl.constexpr,
    stride_an: tl.constexpr,
    stride_am: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_on: tl.constexpr,
    stride_od: tl.constexpr,
    block_m: tl.constexpr,
):
    row = tl.program_id(0)
    dim = tl.program_id(1)
    offsets = tl.arange(0, block_m)
    lam_u = tl.load(lam_u_ptr + row).to(tl.float32)
    res_val = tl.load(res_ptr + row * stride_rn + dim * stride_rd).to(tl.float32)
    acc = 0.0
    for start in range(0, m_ref, block_m):
        cols = start + offsets
        mask = cols < m_ref
        av = tl.load(a_ptr + row * stride_an + cols * stride_am, mask=mask, other=0.0).to(tl.float32)
        bv = tl.load(b_ptr + row * stride_bn + cols * stride_bm, mask=mask, other=0.0).to(tl.float32)
        lv = tl.load(lam_v_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(target_ptr + cols * stride_tm + dim * stride_td, mask=mask, other=0.0).to(tl.float32)
        coeff = lam_u * av + bv * lv
        acc += tl.sum(coeff * (res_val - z), axis=0)
    tl.store(out_ptr + row * stride_on + dim * stride_od, acc / eps_scale)


def triton_cached_ab_implicit_add(
    residuals: torch.Tensor,
    target: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    lam_u: torch.Tensor,
    lam_v: torch.Tensor,
    *,
    epsilon: float,
    cost_scale: float,
    block_m: int = 256,
    num_warps: int = 8,
) -> torch.Tensor:
    """Fused implicit correction for cached A/B PNW backward.

    Computes
    ``sum_j (lam_u_i A_ij + B_ij lam_v_j) * (res_i - target_j) / (eps*scale)``
    without materializing the ``(n, m, d)`` tensor.
    """

    _require_triton()
    tensors = (residuals, target, a, b, lam_u, lam_v)
    if any(t.device.type != "cuda" for t in tensors):
        raise RuntimeError("triton_cached_ab_implicit_add inputs must be CUDA tensors")
    if any(t.dtype != torch.float32 for t in tensors):
        raise TypeError("triton_cached_ab_implicit_add currently supports float32 tensors only")
    residuals = residuals.contiguous()
    target = target.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    lam_u = lam_u.contiguous()
    lam_v = lam_v.contiguous()
    n_rows, d_dim = residuals.shape
    m_ref = target.shape[0]
    if target.shape[1] != d_dim or a.shape != (n_rows, m_ref) or b.shape != (n_rows, m_ref):
        raise ValueError("Incompatible residual, target, A, or B shapes")
    if lam_u.shape != (n_rows,) or lam_v.shape != (m_ref,):
        raise ValueError("Incompatible lambda shapes")
    eps_scale = float(epsilon) * float(cost_scale)
    if eps_scale <= 0 or not math.isfinite(eps_scale):
        raise ValueError("epsilon * cost_scale must be positive and finite")
    out = torch.empty_like(residuals)
    _cached_ab_implicit_add_kernel[(n_rows, d_dim)](
        residuals,
        target,
        a,
        b,
        lam_u,
        lam_v,
        out,
        n_rows,
        m_ref,
        d_dim,
        float(eps_scale),
        residuals.stride(0),
        residuals.stride(1),
        target.stride(0),
        target.stride(1),
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        block_m=int(block_m),
        num_warps=int(num_warps),
    )
    return out
