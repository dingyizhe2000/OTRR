"""Metrics for the paper's simulation and real-data comparisons."""

from .angles import (
    frobenius_angle,
    prediction_space_angle,
    principal_angles,
    weighted_sine_discrepancy,
)
from .errors import (
    coefficient_error,
    centered_prediction_mse,
    mean_bias,
    prediction_mae,
    prediction_mse,
    prediction_mse_per_dim,
    prediction_rmse_per_dim,
    translation_adjusted_mse,
)
from .rotation import rotation_prediction_gap, rotation_prediction_mse_per_dim, rotation_prediction_rmse_per_dim
from .jacobian import jacobian_relative_gap, jacobian_rmse_per_entry, jacobian_rowspace_diagnostics

__all__ = [
    "coefficient_error",
    "centered_prediction_mse",
    "frobenius_angle",
    "mean_bias",
    "jacobian_relative_gap",
    "jacobian_rmse_per_entry",
    "jacobian_rowspace_diagnostics",
    "prediction_mae",
    "prediction_mse",
    "prediction_mse_per_dim",
    "prediction_rmse_per_dim",
    "prediction_space_angle",
    "principal_angles",
    "rotation_prediction_gap",
    "rotation_prediction_mse_per_dim",
    "rotation_prediction_rmse_per_dim",
    "translation_adjusted_mse",
    "weighted_sine_discrepancy",
]
