"""Reference-score samplers for OT-rank."""

from __future__ import annotations

import torch


def _normal_icdf_from_unit_cube(u: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(u.dtype).eps
    clipped = u.clamp(min=eps, max=1.0 - eps)
    return torch.sqrt(torch.tensor(2.0, device=u.device, dtype=u.dtype)) * torch.erfinv(2.0 * clipped - 1.0)


def sample_uniform_ball(
    n: int,
    d: int,
    generator: torch.Generator,
    *,
    radius: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample uniformly from the Euclidean ball in ``R^d``.

    The default reference distribution is centered and spherically symmetric,
    which is the key ingredient behind the orthogonal-equivariance comparisons.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    if d <= 0:
        raise ValueError("d must be positive")
    device = torch.device(device)
    z = torch.randn(n, d, generator=generator, device=device, dtype=dtype)
    norm = torch.linalg.norm(z, dim=1, keepdim=True).clamp_min(torch.finfo(dtype).eps)
    direction = z / norm
    u = torch.rand(n, 1, generator=generator, device=device, dtype=dtype)
    radii = float(radius) * u.pow(1.0 / d)
    return radii * direction


def sample_antipodal_uniform_ball(
    n: int,
    d: int,
    generator: torch.Generator,
    *,
    radius: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample an antipodally paired finite design from the Euclidean ball."""

    if n <= 0:
        raise ValueError("n must be positive")
    if n % 2 != 0:
        raise ValueError("antipodal uniform-ball reference requires an even n")
    half = sample_uniform_ball(
        n // 2,
        d,
        generator,
        radius=radius,
        device=device,
        dtype=dtype,
    )
    return torch.cat([half, -half], dim=0)


def sample_scrambled_sobol_ball(
    n: int,
    d: int,
    *,
    seed: int,
    radius: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate scrambled Sobol reference points in the Euclidean ball.

    The construction draws a scrambled Sobol net in ``[0,1]^(d+1)``. One
    coordinate is transformed to the radius distribution of the uniform ball,
    and the remaining coordinates are transformed through a Gaussian inverse
    CDF and normalized to a direction on the sphere.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    if d <= 0:
        raise ValueError("d must be positive")
    device = torch.device(device)
    engine = torch.quasirandom.SobolEngine(dimension=d + 1, scramble=True, seed=int(seed))
    cube = engine.draw(int(n), dtype=dtype).to(device=device)
    radii = float(radius) * cube[:, :1].clamp_min(torch.finfo(dtype).eps).pow(1.0 / d)
    gaussian = _normal_icdf_from_unit_cube(cube[:, 1:])
    norm = torch.linalg.norm(gaussian, dim=1, keepdim=True).clamp_min(torch.finfo(dtype).eps)
    return radii * (gaussian / norm)


def sample_reference(
    n: int,
    d: int,
    generator: torch.Generator,
    *,
    kind: str = "uniform_ball",
    radius: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample from the unit-ball reference distribution used in the paper."""

    normalized = kind.lower()
    if normalized in {"uniform_ball", "ball"}:
        return sample_uniform_ball(n, d, generator, radius=radius, device=device, dtype=dtype)
    if normalized in {"antipodal_uniform_ball", "antipodal_ball", "antipodal_random_ball"}:
        return sample_antipodal_uniform_ball(n, d, generator, radius=radius, device=device, dtype=dtype)
    if normalized in {"scrambled_sobol_ball", "sobol_ball", "sobol_uniform_ball"}:
        sobol_seed = int(seed if seed is not None else generator.initial_seed())
        return sample_scrambled_sobol_ball(n, d, seed=sobol_seed, radius=radius, device=device, dtype=dtype)
    raise ValueError(f"Unsupported reference distribution: {kind!r}")
