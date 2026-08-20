"""Data generators for OT-rank simulation studies."""

from .noise import NoiseSpec, sample_noise
from .sim import (
    LinearSimulation,
    NonlinearSimulation,
    gaussian_nonlinear_function,
    gaussian_nonlinear_jacobian,
    make_linear_simulation,
    make_nonlinear_simulation,
)
from .download import download_file

__all__ = [
    "LinearSimulation",
    "NonlinearSimulation",
    "NoiseSpec",
    "download_file",
    "gaussian_nonlinear_function",
    "gaussian_nonlinear_jacobian",
    "make_linear_simulation",
    "make_nonlinear_simulation",
    "sample_noise",
]
