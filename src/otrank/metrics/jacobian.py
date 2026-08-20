"""Jacobian-field diagnostics for nonlinear response regressions."""

from __future__ import annotations

import torch


def jacobian_rmse_per_entry(estimated: torch.Tensor, target: torch.Tensor) -> float:
    """Root mean squared difference across test points, responses, and inputs."""

    _validate_jacobians(estimated, target)
    return float(torch.sqrt(torch.mean((estimated - target) ** 2)).item())


def jacobian_relative_gap(estimated: torch.Tensor, target: torch.Tensor) -> float:
    """Frobenius gap normalized by the target Jacobian field norm."""

    _validate_jacobians(estimated, target)
    denominator = torch.linalg.vector_norm(target).clamp_min(torch.finfo(target.dtype).eps)
    return float((torch.linalg.vector_norm(estimated - target) / denominator).item())


def jacobian_rowspace_diagnostics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    """Summarize pointwise row-space angles between two Jacobian fields.

    Principal-angle summaries use points at which both Jacobians have full row
    rank.  Projection distance remains defined at rank-deficient points and
    therefore complements the angle summaries.
    """

    _validate_jacobians(left, right)
    _, d, p = left.shape
    expected_rank = min(d, p)
    _, singular_left, vh_left = torch.linalg.svd(left, full_matrices=False)
    _, singular_right, vh_right = torch.linalg.svd(right, full_matrices=False)
    tolerance_left = 1e-10 * max(d, p) * singular_left[:, :1]
    tolerance_right = 1e-10 * max(d, p) * singular_right[:, :1]
    rank_left = torch.sum(singular_left > tolerance_left, dim=1)
    rank_right = torch.sum(singular_right > tolerance_right, dim=1)
    full_rank = (rank_left == expected_rank) & (rank_right == expected_rank)

    output: dict[str, float] = {
        "full_rank_fraction": float(full_rank.to(left.dtype).mean().item()),
        "rank_mismatch_fraction": float((rank_left != rank_right).to(left.dtype).mean().item()),
    }
    if torch.any(full_rank):
        cross = vh_left[full_rank, :expected_rank] @ vh_right[full_rank, :expected_rank].transpose(-1, -2)
        cosine = torch.linalg.svdvals(cross).clamp(-1.0, 1.0)
        angles = torch.arccos(cosine)
        max_angles = torch.max(angles, dim=1).values
        fro_sin = torch.linalg.vector_norm(torch.sin(angles), dim=1)
        output.update(
            {
                "max_angle_mean": float(torch.mean(max_angles).item()),
                "max_angle_median": float(torch.quantile(max_angles, 0.5).item()),
                "max_angle_q90": float(torch.quantile(max_angles, 0.9).item()),
                "fro_sin_mean": float(torch.mean(fro_sin).item()),
            }
        )
    else:
        output.update(
            {
                "max_angle_mean": float("nan"),
                "max_angle_median": float("nan"),
                "max_angle_q90": float("nan"),
                "fro_sin_mean": float("nan"),
            }
        )

    projection_left = _rowspace_projection(left)
    projection_right = _rowspace_projection(right)
    squared = torch.sum((projection_left - projection_right) ** 2, dim=(-2, -1))
    output["projection_distance"] = float(torch.sqrt(0.5 * torch.mean(squared)).item())
    return output


def _rowspace_projection(jacobian: torch.Tensor) -> torch.Tensor:
    gram = jacobian @ jacobian.transpose(-1, -2)
    return jacobian.transpose(-1, -2) @ torch.linalg.pinv(gram) @ jacobian


def _validate_jacobians(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.ndim != 3 or right.ndim != 3:
        raise ValueError("Jacobian fields must have shape (n, d, p)")
    if left.shape != right.shape:
        raise ValueError(f"Jacobian field shapes differ: {left.shape} versus {right.shape}")
