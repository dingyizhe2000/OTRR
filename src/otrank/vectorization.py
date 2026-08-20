"""Coefficient vectorization conventions for linear-model inference.

The project-wide convention is the standard mathematical column-major
vectorization of a coefficient matrix ``B`` with shape ``(d, p)``::

    vec(B) = (B[0, 0], ..., B[d - 1, 0], B[0, 1], ..., B[d - 1, p - 1]).

NumPy calls this Fortran order.  PyTorch has no order argument, so the same
operation is implemented by transposing the final two axes before reshaping.
Legacy response-major artifacts must be identified explicitly and migrated;
missing metadata is never interpreted implicitly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


COEFFICIENT_VECTORIZATION = "column-major vec(B)"
LEGACY_RESPONSE_MAJOR_VECTORIZATION = "response-major C-order"

_COLUMN_MAJOR_ALIASES = {
    COEFFICIENT_VECTORIZATION,
    "column-major",
    "column_major",
    "fortran",
    "F",
}
_LEGACY_ALIASES = {
    LEGACY_RESPONSE_MAJOR_VECTORIZATION,
    "response-major",
    "response_major",
    "C",
}

__all__ = [
    "COEFFICIENT_VECTORIZATION",
    "LEGACY_RESPONSE_MAJOR_VECTORIZATION",
    "coefficient_index",
    "legacy_response_major_permutation",
    "legacy_response_major_to_column_major_indices",
    "migrate_legacy_influence_rows",
    "migrate_legacy_parameter_vector",
    "migrate_legacy_square_matrix",
    "normalize_vectorization",
    "predictor_indices_for_response",
    "require_vectorization",
    "response_indices_for_predictor",
    "response_left_action_matrix",
    "unvec_coefficient",
    "unvec_coefficient_batch",
    "vec_coefficient",
    "vec_coefficient_batch",
    "vectorization_metadata",
]


def _shape(value: np.ndarray | torch.Tensor) -> tuple[int, ...]:
    return tuple(int(size) for size in value.shape)


def vec_coefficient(
    coefficient: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Return standard column-major ``vec(B)`` for one ``(d, p)`` matrix."""

    if coefficient.ndim != 2:
        raise ValueError("coefficient must have shape (d, p)")
    return coefficient.swapaxes(-2, -1).reshape(-1)


def vec_coefficient_batch(
    coefficients: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Vectorize the final ``(d, p)`` axes of a coefficient-matrix batch."""

    if coefficients.ndim < 2:
        raise ValueError("coefficients must end in shape (d, p)")
    leading = _shape(coefficients)[:-2]
    d, p = _shape(coefficients)[-2:]
    return coefficients.swapaxes(-2, -1).reshape(*leading, p * d)


def unvec_coefficient(
    vector: np.ndarray | torch.Tensor,
    response_dimension: int,
    predictor_dimension: int,
) -> np.ndarray | torch.Tensor:
    """Invert :func:`vec_coefficient` and return a ``(d, p)`` matrix."""

    d = int(response_dimension)
    p = int(predictor_dimension)
    if d < 1 or p < 1:
        raise ValueError("response_dimension and predictor_dimension must be positive")
    size = int(vector.numel() if isinstance(vector, torch.Tensor) else vector.size)
    if vector.ndim != 1 or size != d * p:
        raise ValueError(f"vector must have shape ({d * p},)")
    return vector.reshape(p, d).swapaxes(-2, -1)


def unvec_coefficient_batch(
    vectors: np.ndarray | torch.Tensor,
    response_dimension: int,
    predictor_dimension: int,
) -> np.ndarray | torch.Tensor:
    """Invert :func:`vec_coefficient_batch` on the final axis."""

    d = int(response_dimension)
    p = int(predictor_dimension)
    if d < 1 or p < 1:
        raise ValueError("response_dimension and predictor_dimension must be positive")
    if vectors.ndim < 1 or _shape(vectors)[-1] != d * p:
        raise ValueError(f"vectors must end in shape ({d * p},)")
    leading = _shape(vectors)[:-1]
    return vectors.reshape(*leading, p, d).swapaxes(-2, -1)


def coefficient_index(
    response: int,
    predictor: int,
    response_dimension: int,
    predictor_dimension: int,
) -> int:
    """Return the standard-vectorization index of ``B[response, predictor]``."""

    d = int(response_dimension)
    p = int(predictor_dimension)
    row = int(response)
    column = int(predictor)
    if not 0 <= row < d or not 0 <= column < p:
        raise IndexError("coefficient index is out of bounds")
    return column * d + row


def response_indices_for_predictor(
    predictor: int,
    response_dimension: int,
    predictor_dimension: int,
) -> np.ndarray:
    """Return the contiguous response indices for one predictor column."""

    start = coefficient_index(
        0,
        predictor,
        response_dimension,
        predictor_dimension,
    )
    return np.arange(start, start + int(response_dimension), dtype=np.int64)


def predictor_indices_for_response(
    response: int,
    response_dimension: int,
    predictor_dimension: int,
) -> np.ndarray:
    """Return all predictor-coordinate indices for one response row."""

    d = int(response_dimension)
    p = int(predictor_dimension)
    row = int(response)
    if not 0 <= row < d:
        raise IndexError("response index is out of bounds")
    return row + d * np.arange(p, dtype=np.int64)


def legacy_response_major_to_column_major_indices(
    response_dimension: int,
    predictor_dimension: int,
) -> np.ndarray:
    """Return indices satisfying ``theta_column = theta_legacy[indices]``."""

    d = int(response_dimension)
    p = int(predictor_dimension)
    if d < 1 or p < 1:
        raise ValueError("response_dimension and predictor_dimension must be positive")
    return np.arange(d * p, dtype=np.int64).reshape(d, p).T.reshape(-1)


def legacy_response_major_permutation(
    response_dimension: int,
    predictor_dimension: int,
) -> np.ndarray:
    """Return ``P`` such that ``theta_column = P @ theta_legacy``."""

    indices = legacy_response_major_to_column_major_indices(
        response_dimension,
        predictor_dimension,
    )
    return np.eye(len(indices), dtype=np.float64)[indices]


def migrate_legacy_parameter_vector(
    vector: np.ndarray,
    response_dimension: int,
    predictor_dimension: int,
) -> np.ndarray:
    """Migrate a legacy response-major parameter vector to column-major vec."""

    value = np.asarray(vector)
    indices = legacy_response_major_to_column_major_indices(
        response_dimension,
        predictor_dimension,
    )
    if value.shape != (len(indices),):
        raise ValueError("legacy parameter vector has the wrong shape")
    return value[indices]


def migrate_legacy_influence_rows(
    influence: np.ndarray,
    response_dimension: int,
    predictor_dimension: int,
) -> np.ndarray:
    """Migrate row-wise legacy influence vectors: ``IF_new = IF_old P.T``."""

    value = np.asarray(influence)
    indices = legacy_response_major_to_column_major_indices(
        response_dimension,
        predictor_dimension,
    )
    if value.ndim != 2 or value.shape[1] != len(indices):
        raise ValueError("legacy influence matrix has the wrong shape")
    return value[:, indices]


def migrate_legacy_square_matrix(
    matrix: np.ndarray,
    response_dimension: int,
    predictor_dimension: int,
) -> np.ndarray:
    """Migrate covariance/Hessian matrices: ``M_new = P M_old P.T``."""

    value = np.asarray(matrix)
    indices = legacy_response_major_to_column_major_indices(
        response_dimension,
        predictor_dimension,
    )
    if value.shape != (len(indices), len(indices)):
        raise ValueError("legacy square matrix has the wrong shape")
    return value[np.ix_(indices, indices)]


def response_left_action_matrix(
    response_action: np.ndarray,
    predictor_dimension: int,
) -> np.ndarray:
    """Return the matrix mapping ``vec(B)`` to ``vec(A @ B)``.

    Under column-major vectorization this is ``I_p kron A``.  For a rotated
    fit ``B_Q`` mapped back as ``Q.T @ B_Q``, pass ``response_action=Q.T``.
    """

    action = np.asarray(response_action, dtype=np.float64)
    if action.ndim != 2 or action.shape[0] != action.shape[1]:
        raise ValueError("response_action must be square")
    p = int(predictor_dimension)
    if p < 1:
        raise ValueError("predictor_dimension must be positive")
    return np.kron(np.eye(p, dtype=np.float64), action)


def normalize_vectorization(value: Any) -> str:
    """Normalize a declared vectorization label or reject an unknown one."""

    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    label = str(value)
    if label in _COLUMN_MAJOR_ALIASES:
        return COEFFICIENT_VECTORIZATION
    if label in _LEGACY_ALIASES:
        return LEGACY_RESPONSE_MAJOR_VECTORIZATION
    raise ValueError(f"unknown coefficient vectorization: {label!r}")


def require_vectorization(
    metadata: Mapping[str, Any],
    *,
    allow_legacy: bool = False,
    explicit_legacy_convention: str | None = None,
) -> str:
    """Require artifact vectorization metadata and return its normalized label.

    ``explicit_legacy_convention`` is the only escape hatch for an artifact
    that predates metadata.  Callers must then migrate the loaded arrays with
    the helpers above before using them in current inference code.
    """

    declared: Any | None = None
    if "vectorization" in metadata:
        declared = metadata["vectorization"]
    elif "parameter_order" in metadata:
        declared = metadata["parameter_order"]
    elif explicit_legacy_convention is not None:
        declared = explicit_legacy_convention
    else:
        raise ValueError(
            "artifact has no coefficient vectorization metadata; pass an explicit "
            "legacy convention and migrate it before use"
        )

    normalized = normalize_vectorization(declared)
    if normalized == LEGACY_RESPONSE_MAJOR_VECTORIZATION and not allow_legacy:
        raise ValueError(
            "legacy response-major artifact requires explicit migration to "
            f"{COEFFICIENT_VECTORIZATION!r}"
        )
    return normalized


def vectorization_metadata() -> dict[str, str]:
    """Return metadata fields required on newly written inference artifacts."""

    return {
        "vectorization": COEFFICIENT_VECTORIZATION,
        "coefficient_shape": "(response_dimension, predictor_dimension)",
    }
