"""Utilities for the OT-rank regression paper experiments."""

from .api import (
    LinearOTRankConfig,
    ReLUOTRankConfig,
    as_tensor_pair,
    fit_linear_asm_inference,
    fit_linear_baseline,
    fit_ot_rank_linear,
    fit_ot_rank_relu,
)

__all__ = [
    "LinearOTRankConfig",
    "ReLUOTRankConfig",
    "__version__",
    "as_tensor_pair",
    "fit_linear_asm_inference",
    "fit_linear_baseline",
    "fit_ot_rank_linear",
    "fit_ot_rank_relu",
]

__version__ = "0.1.0"
