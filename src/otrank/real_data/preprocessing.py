"""Shared preprocessing utilities for the NHANES application."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from scipy.linalg import qr

from otrank.estimators.comparison import fit_ols


def common_response_scale(x: torch.Tensor, y: torch.Tensor) -> float:
    """Estimate one robust response scale from OLS residual norms."""

    pilot = fit_ols(x, y)
    residual = y - pilot.predict(x)
    scale = torch.median(torch.linalg.norm(residual, dim=1)) / math.sqrt(y.shape[1])
    return max(float(scale.item()), 1e-8)


def fit_preprocessor(
    train: pd.DataFrame,
    test: pd.DataFrame,
    continuous: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit median imputation and scaling on one training fold."""

    train_out = pd.DataFrame(index=train.index)
    test_out = pd.DataFrame(index=test.index)
    for column in train.columns:
        train_values = pd.to_numeric(train[column], errors="coerce")
        test_values = pd.to_numeric(test[column], errors="coerce")
        if bool(train_values.isna().any()):
            train_out[f"{column}__missing"] = train_values.isna().astype(float)
            test_out[f"{column}__missing"] = test_values.isna().astype(float)
        median = float(train_values.median())
        if not math.isfinite(median):
            raise ValueError(f"Predictor {column!r} is entirely missing in an outer training fold")
        train_values = train_values.fillna(median).astype(float)
        test_values = test_values.fillna(median).astype(float)
        if column in continuous:
            mean = float(train_values.mean())
            std = float(train_values.std(ddof=0))
            if not math.isfinite(std) or std == 0.0:
                std = 1.0
            train_values = (train_values - mean) / std
            test_values = (test_values - mean) / std
        train_out[column] = train_values
        test_out[column] = test_values
    return train_out, test_out


def select_full_rank_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    relative_tolerance: float = 1e-9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove columns that are collinear after accounting for an intercept."""

    values = train.to_numpy(dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    _, triangular, pivots = qr(centered, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(triangular))
    if diagonal.size == 0 or diagonal[0] == 0.0:
        raise ValueError("Predictor design has no nonconstant columns")
    rank = int(np.sum(diagonal > float(relative_tolerance) * diagonal[0]))
    keep = sorted(int(index) for index in pivots[:rank])
    columns = list(train.columns[keep])
    return train.loc[:, columns], test.loc[:, columns]


def _close(raw: np.ndarray) -> np.ndarray:
    out = np.asarray(raw, dtype=np.float64)
    total = out.sum(axis=1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("Each response composition must have positive total abundance")
    return out / total


def _fixed_multiplicative(raw: np.ndarray, delta: float) -> np.ndarray:
    composition = _close(raw)
    zeros = composition <= 0.0
    zero_count = zeros.sum(axis=1)
    if np.any(zero_count * float(delta) >= 1.0):
        raise ValueError("Fixed zero replacement is too large")
    scale = (1.0 - zero_count * float(delta))[:, None]
    out = np.where(zeros, float(delta), composition * scale)
    return out / out.sum(axis=1, keepdims=True)


def _training_replacements(raw_train: np.ndarray, quantile: float) -> np.ndarray:
    composition = _close(raw_train)
    replacements = np.empty(composition.shape[1], dtype=np.float64)
    for column in range(composition.shape[1]):
        positive = composition[:, column][composition[:, column] > 0.0]
        if positive.size == 0:
            raise ValueError(f"Response component {column} has no positive training values")
        replacements[column] = 0.5 * float(np.quantile(positive, quantile))
    return np.clip(replacements, np.finfo(np.float64).tiny, 0.05)


def _component_multiplicative(raw: np.ndarray, replacements: np.ndarray) -> np.ndarray:
    composition = _close(raw)
    zeros = composition <= 0.0
    replacement_mass = (zeros * replacements[None, :]).sum(axis=1)
    if np.any(replacement_mass >= 1.0):
        raise ValueError("Data-driven zero replacement is too large")
    out = np.where(zeros, replacements[None, :], composition)
    positive_mass = np.where(~zeros, composition, 0.0).sum(axis=1)
    positive_scale = np.divide(
        1.0 - replacement_mass,
        positive_mass,
        out=np.ones_like(positive_mass),
        where=positive_mass > 0.0,
    )
    out = np.where(zeros, out, out * positive_scale[:, None])
    return out / out.sum(axis=1, keepdims=True)


def _helmert_basis(components: int) -> np.ndarray:
    basis = np.zeros((components - 1, components), dtype=np.float64)
    for row in range(components - 1):
        width = row + 2
        basis[row, : width - 1] = 1.0 / math.sqrt((width - 1) * width)
        basis[row, width - 1] = -(width - 1) / math.sqrt((width - 1) * width)
    return basis


def transform_response(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    zero_method: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Apply training-defined zero replacement and Helmert ILR coordinates."""

    train = raw_train.to_numpy(dtype=np.float64)
    test = raw_test.to_numpy(dtype=np.float64)
    audit: dict[str, object] = {"zero_method": zero_method}
    if zero_method.startswith("fixed_"):
        delta = float(zero_method.removeprefix("fixed_"))
        train_composition = _fixed_multiplicative(train, delta)
        test_composition = _fixed_multiplicative(test, delta)
        audit["replacement_delta"] = delta
    elif zero_method in {"train_half_min", "train_half_q01", "train_half_q05"}:
        quantile = {
            "train_half_min": 0.0,
            "train_half_q01": 0.01,
            "train_half_q05": 0.05,
        }[zero_method]
        replacements = _training_replacements(train, quantile)
        train_composition = _component_multiplicative(train, replacements)
        test_composition = _component_multiplicative(test, replacements)
        audit["replacement_quantile"] = quantile
        audit["component_replacements"] = replacements.tolist()
    else:
        raise ValueError(f"Unknown zero method: {zero_method}")

    basis = _helmert_basis(train_composition.shape[1])
    return (
        np.log(train_composition) @ basis.T,
        np.log(test_composition) @ basis.T,
        audit,
    )
