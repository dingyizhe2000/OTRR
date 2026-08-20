"""Shared cross-validation utilities for L2-penalized linear baselines."""

from __future__ import annotations

from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Sequence

import numpy as np
import torch

from .baselines import LinearModelFit
from .penalized_linear import (
    LinearASMConfig,
    fit_l2_coordinatewise_asm,
    fit_l2_coordinatewise_huber,
    fit_l2_least_squares,
    fit_l2_radial_huber,
)


PENALIZED_METHODS = (
    "least_squares",
    "coordinatewise_huber",
    "radial_huber",
    "coordinatewise_asm",
)


@dataclass(frozen=True)
class PenaltyCVResult:
    method: str
    selected_penalty: float
    mean_validation_mse: dict[float, float]
    fold_validation_mse: dict[float, tuple[float, ...]]


def fit_penalized_linear_method(
    method: str,
    x: torch.Tensor,
    y: torch.Tensor,
    l2_penalty: float,
    *,
    asm_config: LinearASMConfig | None = None,
) -> LinearModelFit:
    """Fit one method under the common average-loss penalty convention."""

    key = str(method).lower()
    if key == "least_squares":
        return fit_l2_least_squares(x, y, l2_penalty)
    if key == "coordinatewise_huber":
        return fit_l2_coordinatewise_huber(x, y, l2_penalty)
    if key == "radial_huber":
        return fit_l2_radial_huber(x, y, l2_penalty)
    if key == "coordinatewise_asm":
        config = replace(
            asm_config or LinearASMConfig(),
            l2_penalty=float(l2_penalty),
        )
        return fit_l2_coordinatewise_asm(x, y, config)
    raise ValueError(f"Unknown penalized method {method!r}; expected one of {PENALIZED_METHODS}")


def select_l2_penalty(
    method: str,
    x: torch.Tensor,
    y: torch.Tensor,
    folds: Iterable[tuple[Sequence[int], Sequence[int]]],
    penalties: Sequence[float],
    *,
    asm_config: LinearASMConfig | None = None,
    workers: int = 1,
) -> PenaltyCVResult:
    """Select a shared L2 penalty by mean validation MSE on fixed folds."""

    fold_list = [
        (
            torch.as_tensor(train, device=x.device, dtype=torch.long),
            torch.as_tensor(valid, device=x.device, dtype=torch.long),
        )
        for train, valid in folds
    ]
    if not fold_list:
        raise ValueError("At least one CV fold is required")
    penalty_values = tuple(float(value) for value in penalties)
    if not penalty_values or any(value < 0.0 or not np.isfinite(value) for value in penalty_values):
        raise ValueError("penalties must be a nonempty sequence of finite nonnegative values")

    def score_penalty(penalty: float) -> tuple[float, tuple[float, ...]]:
        scores: list[float] = []
        for train_idx, valid_idx in fold_list:
            fit = fit_penalized_linear_method(
                method,
                x[train_idx],
                y[train_idx],
                penalty,
                asm_config=asm_config,
            )
            residual = y[valid_idx] - fit.predict(x[valid_idx])
            score = float(torch.mean(residual.square()).item())
            if not np.isfinite(score):
                raise FloatingPointError(
                    f"Non-finite validation MSE for method={method} penalty={penalty}"
                )
            scores.append(score)
        return penalty, tuple(scores)

    max_workers = max(1, min(int(workers), len(penalty_values)))
    if max_workers == 1:
        scored = [score_penalty(penalty) for penalty in penalty_values]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            scored = list(executor.map(score_penalty, penalty_values))
    fold_scores = dict(scored)

    means = {penalty: float(np.mean(scores)) for penalty, scores in fold_scores.items()}
    selected = min(penalty_values, key=lambda value: (means[value], value))
    return PenaltyCVResult(
        method=str(method),
        selected_penalty=float(selected),
        mean_validation_mse=means,
        fold_validation_mse=fold_scores,
    )
