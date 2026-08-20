"""Noise generator for the paper's linear simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import math

import torch


@dataclass(frozen=True)
class NoiseSpec:
    """Configuration for the low-rank direction-outlier noise used in the paper."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


PAPER_NOISES: dict[str, NoiseSpec] = {
    "A": NoiseSpec(
        "low_rank_direction_outlier",
        {"eps": 0.1, "rank": 1, "scale": 30.0, "sigma": 1.0, "standardize": "global", "tail": "centered_lognormal"},
    ),
    "B": NoiseSpec(
        "low_rank_direction_outlier",
        {"alpha": 2.5, "eps": 0.1, "rank": 3, "scale": 20.0, "standardize": "global", "tail": "centered_pareto"},
    ),
    "C": NoiseSpec(
        "eight_cluster",
        {"scale": 1.0, "sigma": 0.3, "standardize": "global"},
    ),
}


def make_generator(seed: int | None = None, device: torch.device | str = "cpu") -> torch.Generator:
    """Create a PyTorch random generator."""

    gen = torch.Generator(device=str(device))
    if seed is not None:
        gen.manual_seed(int(seed))
    return gen


def random_orthogonal(
    d: int,
    generator: torch.Generator,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Draw a Haar-like random orthogonal matrix using QR decomposition."""

    z = torch.randn(d, d, generator=generator, device=device, dtype=dtype)
    q, r = torch.linalg.qr(z)
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs


def _as_noise_spec(spec: NoiseSpec | dict[str, Any] | str) -> NoiseSpec:
    if isinstance(spec, NoiseSpec):
        return spec
    if isinstance(spec, str):
        if spec not in PAPER_NOISES:
            raise ValueError("Paper simulations only support noise labels A, B, and C")
        return PAPER_NOISES[spec]
    params = dict(spec.get("params", {}))
    return NoiseSpec(str(spec["name"]), params)


def _standardized_lognormal(
    n: int,
    rank: int,
    *,
    sigma: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    raw = torch.exp(float(sigma) * torch.randn(n, rank, generator=generator, device=device, dtype=dtype))
    mean = math.exp(0.5 * float(sigma) ** 2)
    var = (math.exp(float(sigma) ** 2) - 1.0) * math.exp(float(sigma) ** 2)
    return (raw - mean) / math.sqrt(var)


def _standardized_pareto(
    n: int,
    rank: int,
    *,
    alpha: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if float(alpha) <= 2.0:
        raise ValueError("The paper's Pareto noise requires alpha > 2")
    u = torch.rand(n, rank, generator=generator, device=device, dtype=dtype).clamp_min(1e-12)
    raw = u.pow(-1.0 / float(alpha))
    mean = float(alpha) / (float(alpha) - 1.0)
    var = float(alpha) / ((float(alpha) - 1.0) ** 2 * (float(alpha) - 2.0))
    return (raw - mean) / math.sqrt(var)


def _unit_vector(x: torch.Tensor) -> torch.Tensor:
    return x / torch.linalg.norm(x).clamp_min(torch.finfo(x.dtype).eps)


def _empirical_center_global_standardize(noise: torch.Tensor) -> torch.Tensor:
    centered = noise - noise.mean(dim=0, keepdim=True)
    scale = torch.sqrt(torch.mean(centered.square())).clamp_min(torch.finfo(noise.dtype).eps)
    return centered / scale


def _sample_eight_cluster_noise(
    n: int,
    d: int,
    params: dict[str, Any],
    generator: torch.Generator,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    raw = torch.randn(d, 8, generator=generator, device=device, dtype=dtype)
    q = torch.stack([_unit_vector(raw[:, j]) for j in range(8)], dim=1)
    probs = torch.tensor([0.35, 0.20, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02], device=device, dtype=dtype)
    base = torch.zeros(8, d, device=device, dtype=dtype)
    base[0] = q[:, 0]
    base[1] = 1.2 * q[:, 1]
    base[2] = 1.5 * q[:, 2]
    base[3] = 0.8 * _unit_vector(q[:, 0] + q[:, 3])
    base[4] = 1.3 * _unit_vector(q[:, 1] - q[:, 4])
    base[5] = 0.9 * _unit_vector(q[:, 2] + q[:, 5])
    base[6] = 1.1 * _unit_vector(q[:, 3] - q[:, 6])
    base[7] = -sum(probs[j] * base[j] for j in range(7)) / probs[7]
    idx = torch.multinomial(probs, n, replacement=True, generator=generator)
    clusters = base[idx]
    z = float(params.get("sigma", 0.3)) * torch.randn(n, d, generator=generator, device=device, dtype=dtype)
    return z + float(params.get("scale", 1.0)) * clusters


def sample_noise(
    n: int,
    d: int,
    spec: NoiseSpec | dict[str, Any] | str,
    generator: torch.Generator,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Sample the paper's low-rank direction-outlier noise.

    The construction is
    ``xi = g + eta * scale * V a``, where ``g`` is Gaussian, ``eta`` is a
    Bernoulli indicator, the columns of ``V`` are QR-orthonormalized Gaussian
    directions, and the coordinates of ``a`` are standardized centered
    lognormal or Pareto amplitudes. By default, the final simulated sample is
    empirically centered and globally standardized. Exploratory configs may set
    ``standardize="none"`` to keep the population-standardized noise sample
    unnormalized.
    """

    spec = _as_noise_spec(spec)
    name = spec.name.lower().replace("-", "_")
    params = spec.params
    device = torch.device(device)
    standardize = str(params.get("standardize", "global")).lower()

    if name == "eight_cluster":
        x = _sample_eight_cluster_noise(n, d, params, generator, device=device, dtype=dtype)
        if standardize in {"global", "center_global", "centered_global"}:
            return _empirical_center_global_standardize(x)
        if standardize in {"none", "population"}:
            return x
        raise ValueError("standardize must be 'global' or 'none'")

    if name != "low_rank_direction_outlier":
        raise ValueError("Public simulations only support low_rank_direction_outlier and eight_cluster noise")

    eps = float(params.get("eps", 0.1))
    scale = float(params["scale"])
    rank = max(1, min(int(params["rank"]), d))
    tail = str(params["tail"]).lower()

    x = torch.randn(n, d, generator=generator, device=device, dtype=dtype)
    directions_raw = torch.randn(d, rank, generator=generator, device=device, dtype=dtype)
    directions, _ = torch.linalg.qr(directions_raw, mode="reduced")
    mask = torch.rand(n, 1, generator=generator, device=device, dtype=dtype) < eps

    if tail == "centered_lognormal":
        amplitudes = _standardized_lognormal(
            n,
            rank,
            sigma=float(params.get("sigma", 1.0)),
            generator=generator,
            device=device,
            dtype=dtype,
        )
    elif tail == "centered_pareto":
        amplitudes = _standardized_pareto(
            n,
            rank,
            alpha=float(params.get("alpha", 2.5)),
            generator=generator,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError("Paper simulations only support centered_lognormal and centered_pareto amplitudes")

    x = x + mask * scale * (amplitudes @ directions.T)
    if standardize in {"global", "center_global", "centered_global"}:
        return _empirical_center_global_standardize(x)
    if standardize in {"none", "population"}:
        return x
    raise ValueError("standardize must be 'global' or 'none'")
