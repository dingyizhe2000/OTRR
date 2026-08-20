"""PyTorch angle metrics for matrix and prediction comparisons."""

from __future__ import annotations

import math

import torch


def _safe_arccos(x: float) -> float:
    return float(torch.arccos(torch.clamp(torch.as_tensor(x), -1.0, 1.0)).item())


def frobenius_angle(a: torch.Tensor, b: torch.Tensor) -> float:
    """Angle between two matrices after vectorization."""

    denom = torch.linalg.norm(a, ord="fro") * torch.linalg.norm(b, ord="fro")
    if denom <= 1e-15:
        return float("nan")
    return _safe_arccos(float(torch.sum(a * b).item() / denom.item()))


def prediction_space_angle(pred_a: torch.Tensor, pred_b: torch.Tensor) -> float:
    """Frobenius angle between two prediction matrices."""

    denom = torch.linalg.norm(pred_a, ord="fro") * torch.linalg.norm(pred_b, ord="fro")
    if denom <= 1e-15:
        return float("nan")
    return _safe_arccos(float(torch.sum(pred_a * pred_b).item() / denom.item()))


def principal_angles(a: torch.Tensor, b: torch.Tensor, rank: int | None = None) -> torch.Tensor:
    """Principal angles between column spaces of two matrices.

    ``a`` and ``b`` are typically prediction matrices with shape
    ``(n_test, d_response)`` or transposed coefficient matrices. If ``rank`` is
    provided, only the leading left singular vectors are compared.
    """

    ua, sa, _ = torch.linalg.svd(a, full_matrices=False)
    ub, sb, _ = torch.linalg.svd(b, full_matrices=False)
    tol_a = 1e-10 * max(a.shape) * (sa[0] if sa.numel() else torch.tensor(0.0, device=a.device, dtype=a.dtype))
    tol_b = 1e-10 * max(b.shape) * (sb[0] if sb.numel() else torch.tensor(0.0, device=b.device, dtype=b.dtype))
    ra = int(torch.sum(sa > tol_a).item())
    rb = int(torch.sum(sb > tol_b).item())
    r = min(ra, rb)
    if rank is not None:
        r = min(r, rank)
    if r == 0:
        return torch.empty(0, device=a.device, dtype=a.dtype)
    qa = ua[:, :r]
    qb = ub[:, :r]
    singular = torch.linalg.svdvals(qa.T @ qb)
    return torch.arccos(torch.clamp(singular, -1.0, 1.0))


def _rowspace_basis(
    matrix: torch.Tensor,
    *,
    rtol: float | None,
    atol: float,
) -> torch.Tensor:
    """Return an orthonormal row-space basis using a numerical SVD rank."""

    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    if singular_values.numel() == 0:
        return vh[:0]
    relative_tolerance = (
        max(matrix.shape) * torch.finfo(matrix.dtype).eps
        if rtol is None
        else rtol
    )
    tolerance = max(atol, relative_tolerance * float(singular_values[0].item()))
    rank = int(torch.count_nonzero(singular_values > tolerance).item())
    return vh[:rank]


def weighted_sine_discrepancy(
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    rtol: float | None = None,
    atol: float = 0.0,
    return_angle_degrees: bool = False,
) -> float | tuple[float, float]:
    """Compute the symmetric, coefficient-energy-weighted sine discrepancy.

    For row-space projectors ``P_b`` and ``P_c``, the returned discrepancy is

    ``D = sqrt(0.5 * (||B(I-P_c)||_F^2 / ||B||_F^2
                     + ||C(I-P_b)||_F^2 / ||C||_F^2))``.

    Numerical row ranks are selected by SVD with threshold
    ``max(atol, rtol * sigma_max)``. When ``rtol`` is omitted, the standard
    dtype-aware threshold ``max(shape) * eps * sigma_max`` is used. Both
    matrices must be finite, nonzero, real floating-point tensors with the
    same number of columns. If ``return_angle_degrees`` is true, return
    ``(D, asin(D) * 180 / pi)``.
    """

    if not isinstance(b, torch.Tensor) or not isinstance(c, torch.Tensor):
        raise TypeError("b and c must be torch tensors")
    if b.ndim != 2 or c.ndim != 2:
        raise ValueError("b and c must be two-dimensional matrices")
    if b.shape[1] != c.shape[1]:
        raise ValueError("b and c must have the same number of columns")
    if b.device != c.device:
        raise ValueError("b and c must be on the same device")
    if (
        not b.is_floating_point()
        or not c.is_floating_point()
        or b.is_complex()
        or c.is_complex()
    ):
        raise TypeError("b and c must be real floating-point tensors")
    if rtol is not None and (not math.isfinite(rtol) or rtol < 0.0):
        raise ValueError("rtol must be finite and nonnegative")
    if not math.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and nonnegative")

    dtype = torch.promote_types(b.dtype, c.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    b_work = b.to(dtype=dtype)
    c_work = c.to(dtype=dtype)
    if not torch.isfinite(b_work).all() or not torch.isfinite(c_work).all():
        raise ValueError("b and c must contain only finite values")

    b_energy = torch.sum(b_work.square())
    c_energy = torch.sum(c_work.square())
    if float(b_energy.item()) == 0.0 or float(c_energy.item()) == 0.0:
        raise ValueError("weighted sine discrepancy is undefined for a zero matrix")

    basis_b = _rowspace_basis(b_work, rtol=rtol, atol=atol)
    basis_c = _rowspace_basis(c_work, rtol=rtol, atol=atol)

    residual_b = b_work - (b_work @ basis_c.T) @ basis_c
    residual_c = c_work - (c_work @ basis_b.T) @ basis_b
    term_b = torch.sum(residual_b.square()) / b_energy
    term_c = torch.sum(residual_c.square()) / c_energy
    discrepancy_squared = torch.clamp(
        0.5 * (term_b + term_c),
        min=0.0,
        max=1.0,
    )
    discrepancy = float(torch.sqrt(discrepancy_squared).item())

    if not return_angle_degrees:
        return discrepancy
    angle_degrees = math.degrees(math.asin(min(1.0, max(0.0, discrepancy))))
    return discrepancy, angle_degrees
