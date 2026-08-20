"""Estimators used in simulations."""

from .baselines import LinearModelFit, fit_coordinatewise_huber, fit_ols, fit_radial_huber, fit_zero_intercept_ols
from .comparison import (
    ASMConfig,
    ASMFitInferenceResult,
    MCQROTConfig,
    fit_coordinatewise_asm,
    fit_coordinatewise_asm_with_inference,
    fit_mcqr_ot_lens,
)
from .cw_cqr import CWCQRInference, cqr_significance_summary, cw_cqr_covariance
from .cw_cqr_bridge import (
    CW_CQR_METHOD_ID,
    CWCQRFit,
    CWCQRFitConfig,
    equally_spaced_quantiles,
    fit_cw_cqr,
    load_cw_cqr_fit,
)
from .neural_pnw import FullRiskPNWReLUConfig, FullRiskPNWReLUEstimator, ReLUFeedForward, ReLUFeedForwardFit
from .neural_baselines import (
    AntitonicScore,
    ReLUASMConfig,
    ReLUCQRConfig,
    ReLURegressionConfig,
    estimate_antitonic_score,
    fit_relu_coordinatewise_asm,
    fit_relu_coordinatewise_cqr,
    fit_relu_coordinatewise_huber,
    fit_relu_least_squares,
    fit_relu_radial_huber,
)
from .pnw_fullrisk import FullRiskPNWLinearConfig, FullRiskPNWLinearEstimator
from .penalized_linear import (
    LinearASMConfig,
    fit_l2_coordinatewise_asm,
    fit_l2_coordinatewise_huber,
    fit_l2_least_squares,
    fit_l2_radial_huber,
)
from .penalized_cv import (
    PENALIZED_METHODS,
    PenaltyCVResult,
    fit_penalized_linear_method,
    select_l2_penalty,
)

__all__ = [
    "ASMConfig",
    "ASMFitInferenceResult",
    "CW_CQR_METHOD_ID",
    "CWCQRFit",
    "CWCQRFitConfig",
    "CWCQRInference",
    "FullRiskPNWLinearConfig",
    "FullRiskPNWLinearEstimator",
    "FullRiskPNWReLUConfig",
    "FullRiskPNWReLUEstimator",
    "LinearModelFit",
    "LinearASMConfig",
    "PENALIZED_METHODS",
    "PenaltyCVResult",
    "MCQROTConfig",
    "ReLUFeedForward",
    "ReLUFeedForwardFit",
    "AntitonicScore",
    "ReLUASMConfig",
    "ReLUCQRConfig",
    "ReLURegressionConfig",
    "estimate_antitonic_score",
    "fit_coordinatewise_huber",
    "fit_cw_cqr",
    "fit_l2_coordinatewise_asm",
    "fit_l2_coordinatewise_huber",
    "fit_l2_least_squares",
    "fit_l2_radial_huber",
    "fit_penalized_linear_method",
    "fit_coordinatewise_asm",
    "fit_coordinatewise_asm_with_inference",
    "fit_mcqr_ot_lens",
    "fit_ols",
    "fit_radial_huber",
    "fit_relu_coordinatewise_asm",
    "fit_relu_coordinatewise_cqr",
    "fit_relu_coordinatewise_huber",
    "fit_relu_least_squares",
    "fit_relu_radial_huber",
    "fit_zero_intercept_ols",
    "cqr_significance_summary",
    "cw_cqr_covariance",
    "equally_spaced_quantiles",
    "load_cw_cqr_fit",
    "select_l2_penalty",
]
