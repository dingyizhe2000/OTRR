"""Simulation data-generating processes built on PyTorch tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import torch

from .noise import NoiseSpec, make_generator, sample_noise


@dataclass
class LinearSimulation:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    f_train: torch.Tensor
    f_test: torch.Tensor
    coef: torch.Tensor
    noise_train: torch.Tensor
    noise_test: torch.Tensor


@dataclass
class NonlinearSimulation:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    f_train: torch.Tensor
    f_test: torch.Tensor
    noise_train: torch.Tensor
    noise_test: torch.Tensor


def make_linear_coef(
    d: int,
    p: int,
    generator: torch.Generator,
    rank: int | None = None,
    signal: float = 1.0,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Create a coefficient matrix with controlled Frobenius norm."""

    if rank is None or rank >= min(d, p):
        coef = torch.randn(d, p, generator=generator, device=device, dtype=dtype)
    else:
        left = torch.randn(d, rank, generator=generator, device=device, dtype=dtype)
        right = torch.randn(rank, p, generator=generator, device=device, dtype=dtype)
        coef = left @ right
    norm = torch.linalg.norm(coef, ord="fro")
    if norm > 0:
        coef = signal * coef / norm * math.sqrt(d)
    return coef


def make_covariates(
    n: int,
    p: int,
    generator: torch.Generator,
    rho: float = 0.0,
    *,
    center: bool = True,
    mu_nonzero: int = 0,
    mu_value: float = 0.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Generate Gaussian covariates with optional AR(1) correlation."""

    device = torch.device(device)
    z = torch.randn(n, p, generator=generator, device=device, dtype=dtype)
    if abs(rho) > 1e-12:
        idx = torch.arange(p, device=device, dtype=dtype)
        corr = rho ** torch.abs(idx[:, None] - idx[None, :])
        eigvals, eigvecs = torch.linalg.eigh(corr)
        root = eigvecs @ torch.diag(torch.sqrt(torch.clamp(eigvals, min=1e-8))) @ eigvecs.T
        z = z @ root.T
    if center:
        z = z - z.mean(dim=0, keepdim=True)
    if int(mu_nonzero) > 0 and abs(float(mu_value)) > 1e-12:
        mu = torch.zeros(p, device=device, dtype=dtype)
        mu[: min(int(mu_nonzero), p)] = float(mu_value)
        z = z + mu
    return z


def make_decay_linear_coef(
    d: int,
    p: int,
    generator: torch.Generator,
    *,
    decay: float = 0.7,
    signal: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Create ``B0 = U diag(decay^j) V^T`` with ``||B0||_F = signal sqrt(d)``."""

    raw_u = torch.randn(d, d, generator=generator, device=device, dtype=dtype)
    u, r_u = torch.linalg.qr(raw_u)
    signs_u = torch.sign(torch.diagonal(r_u))
    signs_u = torch.where(signs_u == 0, torch.ones_like(signs_u), signs_u)
    u = u * signs_u
    raw_v = torch.randn(p, d, generator=generator, device=device, dtype=dtype)
    v, _ = torch.linalg.qr(raw_v, mode="reduced")
    singular = torch.tensor([float(decay) ** j for j in range(d)], device=device, dtype=dtype)
    coef = u @ torch.diag(singular) @ v.T
    norm = torch.linalg.norm(coef, ord="fro").clamp_min(1e-12)
    return float(signal) * coef / norm * math.sqrt(d)


def make_linear_simulation(
    *,
    n_train: int,
    n_test: int,
    p: int,
    d: int,
    noise: NoiseSpec | dict[str, Any] | str,
    seed: int,
    x_rho: float = 0.0,
    x_center: bool = True,
    mu_nonzero: int = 0,
    mu_value: float = 0.0,
    coef_rank: int | None = None,
    coef_style: str = "gaussian",
    singular_decay: float = 0.7,
    signal: float = 1.0,
    noise_scale: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> LinearSimulation:
    """Generate data from ``Y = B X + noise``."""

    generator = make_generator(seed, device=device)
    style = str(coef_style).lower()
    if style in {"decay", "decaying_singular_values"}:
        coef = make_decay_linear_coef(
            d,
            p,
            generator,
            decay=float(singular_decay),
            signal=signal,
            device=device,
            dtype=dtype,
        )
    elif style in {"gaussian", "random"}:
        coef = make_linear_coef(d, p, generator, rank=coef_rank, signal=signal, device=device, dtype=dtype)
    else:
        raise ValueError("coef_style must be 'gaussian' or 'decay'")
    x_train = make_covariates(
        n_train,
        p,
        generator,
        rho=x_rho,
        center=x_center,
        mu_nonzero=mu_nonzero,
        mu_value=mu_value,
        device=device,
        dtype=dtype,
    )
    x_test = make_covariates(
        n_test,
        p,
        generator,
        rho=x_rho,
        center=x_center,
        mu_nonzero=mu_nonzero,
        mu_value=mu_value,
        device=device,
        dtype=dtype,
    )
    f_train = x_train @ coef.T
    f_test = x_test @ coef.T
    noise_train = noise_scale * sample_noise(n_train, d, noise, generator, device=device, dtype=dtype)
    noise_test = noise_scale * sample_noise(n_test, d, noise, generator, device=device, dtype=dtype)
    y_train = f_train + noise_train
    y_test = f_test + noise_test
    return LinearSimulation(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        f_train=f_train,
        f_test=f_test,
        coef=coef,
        noise_train=noise_train,
        noise_test=noise_test,
    )


def _project4_features(x: torch.Tensor) -> torch.Tensor:
    """Smooth nonlinear features adapted from project 4's simulation function."""

    if x.shape[1] < 8:
        raise ValueError("project4 nonlinear DGP requires p >= 8")
    z = torch.sigmoid(x)
    h1 = torch.cos(2.0 * math.pi * z[:, 0])
    h2 = torch.sigmoid(z[:, 1] + z[:, 2])
    h3 = 1.0 / (1.0 + z[:, 3] + z[:, 4]).pow(3.0)
    h4 = 1.0 / (0.25 + z[:, 5] + torch.exp(z[:, 6] * z[:, 7]))
    h5 = torch.sin(0.5 * math.pi * (z[:, 0] + z[:, 1]))
    h6 = torch.log1p((z[:, 2] * z[:, 3] * z[:, 4]).pow(2.0))
    h7 = z[:, 7] * torch.sigmoid(z[:, 5] + z[:, 6])
    h8 = h1 * h2 + h3 - h4
    return torch.stack([h1, h2, h3, h4, h5, h6, h7, h8], dim=1)


# Population normalization under X ~ N(0, I_8).  These constants were
# computed once from a fixed 2^22-point scrambled Gaussian QMC design.  Keeping
# them fixed is important: every Monte Carlo repetition represents the same
# regression function, rather than a sample-standardized version of it.
GAUSSIAN_NONLINEAR_MEAN = (0.121305293061, 0.0, 0.184963209067, 0.472929295676)
GAUSSIAN_NONLINEAR_STD = (0.692672261727, 0.539981585121, 0.695816052162, 0.523647380146)


def gaussian_nonlinear_raw_function(x: torch.Tensor) -> torch.Tensor:
    """Four genuinely nonlinear response coordinates for Gaussian covariates."""

    if x.ndim != 2 or x.shape[1] != 8:
        raise ValueError("Gaussian nonlinear DGP requires x with shape (n, 8)")
    x1, x2, x3, x4, x5, x6, x7, x8 = x.unbind(dim=1)
    g1 = (
        torch.sin(x1 * x2 + 0.5 * x3)
        + 0.6 * torch.tanh(x4 * x5)
        + 0.4 * torch.exp(-0.5 * (x6.square() + x7.square())) * torch.cos(x8)
    )
    g2 = (
        torch.cos(x1 + x4 * x6) * torch.tanh(x2 - x7)
        + 0.5 * torch.sigmoid(x3 * x5 + x8)
        - 0.25
    )
    g3 = (
        torch.tanh(x1 * x7 + torch.sin(x2))
        + 0.5 * torch.cos(x3 * x4) * torch.exp(-torch.abs(x5))
        + 0.4 * (x6 * x8) / torch.sqrt(1.0 + (x6 * x8).square())
    )
    g4 = (
        torch.sin(x1 + x8) * torch.sigmoid(x2 * x6)
        + 0.5 * torch.tanh(x3.square() - x7)
        + 0.4 * torch.cos(x4 * x5)
    )
    return torch.stack((g1, g2, g3, g4), dim=1)


def gaussian_nonlinear_function(x: torch.Tensor, *, signal: float = 1.0) -> torch.Tensor:
    """Population-standardized nonlinear regression function."""

    raw = gaussian_nonlinear_raw_function(x)
    mean = raw.new_tensor(GAUSSIAN_NONLINEAR_MEAN)
    scale = raw.new_tensor(GAUSSIAN_NONLINEAR_STD)
    return float(signal) * (raw - mean) / scale


def gaussian_nonlinear_jacobian(x: torch.Tensor, *, signal: float = 1.0) -> torch.Tensor:
    """Jacobian of the fixed nonlinear regression function at each input row."""

    inputs = x.detach().clone().requires_grad_(True)
    outputs = gaussian_nonlinear_function(inputs, signal=signal)
    gradients = [
        torch.autograd.grad(outputs[:, j].sum(), inputs, retain_graph=j + 1 < outputs.shape[1])[0]
        for j in range(outputs.shape[1])
    ]
    return torch.stack(gradients, dim=1).detach()


def make_project4_multivariate_function(
    x: torch.Tensor,
    d: int,
    generator: torch.Generator,
    *,
    signal: float = 1.0,
    coef: torch.Tensor | None = None,
    standardize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a multivariate smooth nonlinear regression function.

    The scalar building blocks follow project 4's simulation design. Multiple
    response coordinates are obtained by a fixed random mixing matrix, so the
    true function is not itself one of the fitted neural-network models.
    """

    features = _project4_features(x)
    if coef is None:
        coef = torch.randn(features.shape[1], d, generator=generator, device=x.device, dtype=x.dtype)
    else:
        coef = coef.to(device=x.device, dtype=x.dtype).T if coef.shape[0] == d else coef.to(device=x.device, dtype=x.dtype)
    coef = coef / torch.linalg.norm(coef, dim=0, keepdim=True).clamp_min(1e-8)
    f = features @ coef
    if standardize:
        f = f - f.mean(dim=0, keepdim=True)
        scale = f.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        f = float(signal) * f / scale
    return f, coef.T


def make_nonlinear_simulation(
    *,
    n_train: int,
    n_test: int,
    p: int,
    d: int,
    noise: NoiseSpec | dict[str, Any] | str,
    seed: int,
    x_rho: float = 0.0,
    signal: float = 1.0,
    noise_scale: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> NonlinearSimulation:
    """Generate the fixed four-response nonlinear Gaussian simulation."""

    if int(p) != 8 or int(d) != 4:
        raise ValueError("The fixed Gaussian nonlinear DGP requires (p, d) = (8, 4)")

    generator = make_generator(seed, device=device)
    x_train = make_covariates(n_train, p, generator, rho=x_rho, center=False, device=device, dtype=dtype)
    x_test = make_covariates(n_test, p, generator, rho=x_rho, center=False, device=device, dtype=dtype)
    f_train = gaussian_nonlinear_function(x_train, signal=signal)
    f_test = gaussian_nonlinear_function(x_test, signal=signal)
    noise_train = noise_scale * sample_noise(n_train, d, noise, generator, device=device, dtype=dtype)
    noise_test = noise_scale * sample_noise(n_test, d, noise, generator, device=device, dtype=dtype)
    return NonlinearSimulation(
        x_train=x_train,
        y_train=f_train + noise_train,
        x_test=x_test,
        y_test=f_test + noise_test,
        f_train=f_train,
        f_test=f_test,
        noise_train=noise_train,
        noise_test=noise_test,
    )
