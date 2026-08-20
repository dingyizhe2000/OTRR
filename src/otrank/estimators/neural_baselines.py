"""Least-squares, coordinatewise Huber, and ASM ReLU regressions."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable

import numpy as np
import torch

from .neural_pnw import ReLUFeedForward, ReLUFeedForwardFit


@dataclass
class ReLURegressionConfig:
    """Optimization settings shared by the ReLU comparison estimators."""

    width: int = 16
    hidden_layers: int = 3
    steps: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-3
    optimizer: str = "adam"
    momentum: float = 0.0
    output_init: str = "default"
    intercept_mode: str = "center"
    seed: int = 20260721
    huber_tau: float = 1.345
    radial_huber_tau: float | None = None
    radial_huber_tau_multiplier: float = 1.345
    trace_every: int = 0


@dataclass
class ReLUASMConfig:
    """Coordinatewise antitonic score matching for a ReLU candidate class."""

    pilot: ReLURegressionConfig = field(
        default_factory=lambda: ReLURegressionConfig(steps=1000, lr=1e-3, weight_decay=1e-3)
    )
    width: int = 16
    hidden_layers: int = 3
    steps: int = 1000
    lr: float = 5e-4
    weight_decay: float = 1e-3
    optimizer: str = "adam"
    momentum: float = 0.0
    alt_iter: int = 1
    score_grid_size: int = 3000
    kernel_points: int = 32768
    bandwidth: str | float = "nrd0"
    seed: int = 20260721
    trace_every: int = 0


@dataclass
class ReLUCQRConfig:
    """Coordinatewise composite-quantile loss for a shared ReLU network."""

    width: int = 16
    hidden_layers: int = 3
    steps: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-3
    quantile_levels: tuple[float, ...] = tuple(level / 21.0 for level in range(1, 21))
    output_init: str = "default"
    intercept_mode: str = "center"
    seed: int = 20260721


@dataclass(frozen=True)
class AntitonicScore:
    knots: np.ndarray
    values: np.ndarray
    bandwidth: float


def _pava_decreasing(values: np.ndarray) -> np.ndarray:
    """Equal-weight decreasing PAVA, expanded to the input length."""

    levels: list[float] = []
    weights: list[int] = []
    for value in np.asarray(values, dtype=np.float64):
        levels.append(float(value))
        weights.append(1)
        while len(levels) >= 2 and levels[-2] < levels[-1]:
            total = weights[-2] + weights[-1]
            pooled = (weights[-2] * levels[-2] + weights[-1] * levels[-1]) / total
            levels[-2:] = [pooled]
            weights[-2:] = [total]
    return np.concatenate([np.full(weight, level) for level, weight in zip(levels, weights, strict=True)])


def _nrd0_bandwidth(x: np.ndarray) -> float:
    n = int(x.size)
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    iqr = float(np.subtract(*np.percentile(x, [75.0, 25.0]))) / 1.34
    candidates = [value for value in (std, iqr) if math.isfinite(value) and value > 0]
    base = min(candidates) if candidates else max(abs(float(np.mean(x))), 1.0)
    return max(0.9 * base * n ** (-0.2), np.finfo(np.float64).eps)


def estimate_antitonic_score(
    residuals: np.ndarray | torch.Tensor,
    *,
    score_grid_size: int = 3000,
    kernel_points: int = 32768,
    bandwidth: str | float = "nrd0",
) -> AntitonicScore:
    """Match ``asm 0.2.4`` Gaussian-KDE score projection in Python."""

    if isinstance(residuals, torch.Tensor):
        sample = residuals.detach().cpu().double().numpy().reshape(-1)
    else:
        sample = np.asarray(residuals, dtype=np.float64).reshape(-1)
    sample = np.sort(sample[np.isfinite(sample)])
    if sample.size < 3:
        raise ValueError("ASM score estimation requires at least three finite residuals")
    bw = _nrd0_bandwidth(sample) if isinstance(bandwidth, str) else float(bandwidth)
    if not math.isfinite(bw) or bw <= 0:
        raise ValueError("ASM bandwidth must be positive and finite")

    support1 = np.array([sample[0] - 3.0 * bw, sample[-1] + 3.0 * bw])
    median = float(np.median(sample))
    raw_iqr = float(np.subtract(*np.percentile(sample, [75.0, 25.0])))
    support2 = np.array([median - 20.0 * raw_iqr, median + 20.0 * raw_iqr])
    lo, hi = max(support1[0], support2[0]), min(support1[1], support2[1])
    if not hi > lo:
        lo, hi = support1
    grid = np.linspace(lo, hi, max(512, int(kernel_points)), dtype=np.float64)

    density = np.zeros_like(grid)
    normalizer = sample.size * bw * math.sqrt(2.0 * math.pi)
    chunk = max(1, min(sample.size, 256))
    for start in range(0, sample.size, chunk):
        z = (grid[:, None] - sample[None, start : start + chunk]) / bw
        density += np.exp(-0.5 * z * z).sum(axis=1)
    density /= normalizer

    masses = 0.5 * (density[1:] + density[:-1]) * np.diff(grid)
    cdf = np.concatenate(([0.0], np.cumsum(masses)))
    if not cdf[-1] > 0:
        raise FloatingPointError("KDE has zero integrated mass")
    cdf /= cdf[-1]

    n = sample.size
    k = max(int(score_grid_size), 2 * n)
    skip = 2 * (k // n) if n > 2 else 0
    u = np.linspace(0.0, 1.0, k + 1)
    quantiles = np.interp(u, cdf, grid)
    density_quantiles = np.interp(quantiles, grid, density)
    initial_scores = np.diff(density_quantiles) * k
    values = _pava_decreasing(initial_scores)
    domain = 0.5 * (quantiles[1:] + quantiles[:-1])
    stop = k - skip
    domain = domain[skip:stop]
    values = values[skip:stop]
    if domain.size < 2:
        raise FloatingPointError("ASM score projection retained fewer than two knots")

    jumps = np.diff(values) != 0
    keep = np.concatenate((jumps, [True])) | np.concatenate(([True], jumps))
    domain = domain[keep]
    values = values[keep]
    values[0] = max(values[0], 1e-7)
    values[-1] = min(values[-1], -1e-7)
    return AntitonicScore(knots=domain, values=values, bandwidth=bw)


class _IntegratedScoreLoss(torch.nn.Module):
    """Negative antiderivative of a decreasing piecewise-linear score."""

    def __init__(self, score: AntitonicScore, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        knots = torch.as_tensor(score.knots, device=device, dtype=dtype)
        values = torch.as_tensor(score.values, device=device, dtype=dtype)
        dx = knots[1:] - knots[:-1]
        cumulative = torch.cat((knots.new_zeros(1), torch.cumsum(0.5 * (values[1:] + values[:-1]) * dx, 0)))
        zero_integral = self._integral_at(knots.new_tensor(0.0), knots, values, cumulative)
        self.register_buffer("knots", knots)
        self.register_buffer("values", values)
        self.register_buffer("cumulative", cumulative - zero_integral)

    @staticmethod
    def _integral_at(x: torch.Tensor, knots: torch.Tensor, values: torch.Tensor, cumulative: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1).contiguous()
        idx = torch.searchsorted(knots, flat, right=True) - 1
        left_tail = idx < 0
        right_tail = idx >= knots.numel() - 1
        middle = idx.clamp(0, knots.numel() - 2)
        x0 = knots[middle]
        dx = knots[middle + 1] - x0
        slope = (values[middle + 1] - values[middle]) / dx
        offset = flat - x0
        result = cumulative[middle] + values[middle] * offset + 0.5 * slope * offset.square()
        result = torch.where(left_tail, cumulative[0] + values[0] * (flat - knots[0]), result)
        result = torch.where(right_tail, cumulative[-1] + values[-1] * (flat - knots[-1]), result)
        return result.reshape(x.shape)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return -self._integral_at(residual, self.knots, self.values, self.cumulative)


def _center_data(x: torch.Tensor, y: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode.lower() == "zero":
        x_mean = x.new_zeros(1, x.shape[1])
        y_mean = y.new_zeros(1, y.shape[1])
    elif mode.lower() in {"center", "intercept"}:
        x_mean = x.mean(0, keepdim=True)
        y_mean = y.mean(0, keepdim=True)
    else:
        raise ValueError("intercept_mode must be 'zero', 'center', or 'intercept'")
    return x - x_mean, y - y_mean, x_mean, y_mean


def _fit_relu_objective(
    x: torch.Tensor,
    y: torch.Tensor,
    config: ReLURegressionConfig,
    objective: Callable[[torch.Tensor], torch.Tensor],
    *,
    method: str,
    init_fit: ReLUFeedForwardFit | None = None,
) -> ReLUFeedForwardFit:
    xc, yc, x_mean, y_mean = _center_data(x, y, config.intercept_mode)
    torch.manual_seed(int(config.seed))
    model = ReLUFeedForward(x.shape[1], y.shape[1], width=config.width, hidden_layers=config.hidden_layers).to(x)
    if init_fit is not None:
        model.load_state_dict({key: value.to(x) for key, value in init_fit.model_state.items()})
    else:
        output_init = str(config.output_init).lower()
        if output_init == "zero":
            output_layer = model.net[-1]
            if not isinstance(output_layer, torch.nn.Linear):
                raise TypeError("ReLUFeedForward must end in a linear output layer")
            torch.nn.init.zeros_(output_layer.weight)
            torch.nn.init.zeros_(output_layer.bias)
        elif output_init != "default":
            raise ValueError("output_init must be 'default' or 'zero'")

    optimizer_name = str(config.optimizer).lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(config.lr),
            momentum=float(config.momentum),
            weight_decay=float(config.weight_decay),
        )
    else:
        raise ValueError("optimizer must be 'adam' or 'sgd'")
    for step in range(1, int(config.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        residual = yc - model(xc)
        loss = objective(residual).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {method} loss at step {step}")
        loss.backward()
        optimizer.step()
    return ReLUFeedForwardFit(
        model_state={key: value.detach().clone() for key, value in model.state_dict().items()},
        input_dim=x.shape[1],
        output_dim=y.shape[1],
        width=config.width,
        hidden_layers=config.hidden_layers,
        x_mean=x_mean,
        y_mean=y_mean,
        method=method,
    )


def fit_relu_least_squares(x: torch.Tensor, y: torch.Tensor, config: ReLURegressionConfig | None = None) -> ReLUFeedForwardFit:
    config = config or ReLURegressionConfig()
    return _fit_relu_objective(x, y, config, lambda residual: 0.5 * residual.square(), method="relu_least_squares")


def fit_relu_coordinatewise_huber(
    x: torch.Tensor,
    y: torch.Tensor,
    config: ReLURegressionConfig | None = None,
) -> ReLUFeedForwardFit:
    config = config or ReLURegressionConfig()
    _, yc, _, _ = _center_data(x, y, config.intercept_mode)
    median = yc.median(dim=0).values
    scale = (yc - median).abs().median(dim=0).values / 0.6745
    cutoff = float(config.huber_tau) * scale.clamp_min(1e-6)

    def objective(residual: torch.Tensor) -> torch.Tensor:
        absolute = residual.abs()
        return torch.where(absolute <= cutoff, 0.5 * residual.square(), cutoff * (absolute - 0.5 * cutoff))

    return _fit_relu_objective(x, y, config, objective, method="relu_coordinatewise_huber")


def fit_relu_radial_huber(
    x: torch.Tensor,
    y: torch.Tensor,
    config: ReLURegressionConfig | None = None,
) -> ReLUFeedForwardFit:
    """Fit ReLU regression with a rotation-invariant radial Huber loss."""

    config = config or ReLURegressionConfig()
    cutoff = (
        float(config.radial_huber_tau)
        if config.radial_huber_tau is not None
        else float(config.radial_huber_tau_multiplier) * math.sqrt(y.shape[1])
    )
    if cutoff <= 0:
        raise ValueError("radial Huber cutoff must be positive")

    def objective(residual: torch.Tensor) -> torch.Tensor:
        radius = torch.linalg.vector_norm(residual, dim=1)
        return torch.where(radius <= cutoff, 0.5 * radius.square(), cutoff * (radius - 0.5 * cutoff))

    return _fit_relu_objective(x, y, config, objective, method="relu_radial_huber")


def fit_relu_coordinatewise_cqr(
    x: torch.Tensor,
    y: torch.Tensor,
    config: ReLUCQRConfig | None = None,
) -> ReLUFeedForwardFit:
    """Fit a ReLU mean function with responsewise composite-quantile losses.

    Quantile-specific intercepts are nuisance parameters. Their average is
    constrained to zero while optimizing, and the fitted function is finally
    shifted so that its training residuals have zero coordinatewise mean. This
    separates the reported conditional-mean predictor from asymmetric error
    quantiles without changing the fitted composite-quantile objective.
    """

    config = config or ReLUCQRConfig()
    levels = torch.as_tensor(config.quantile_levels, device=x.device, dtype=x.dtype)
    if levels.ndim != 1 or levels.numel() == 0:
        raise ValueError("quantile_levels must be a nonempty one-dimensional sequence")
    if not bool(torch.all((levels > 0) & (levels < 1))):
        raise ValueError("all quantile levels must lie strictly between zero and one")
    if not bool(torch.all(levels[1:] > levels[:-1])):
        raise ValueError("quantile_levels must be strictly increasing")

    xc, yc, x_mean, y_mean = _center_data(x, y, config.intercept_mode)
    torch.manual_seed(int(config.seed))
    model = ReLUFeedForward(
        x.shape[1],
        y.shape[1],
        width=int(config.width),
        hidden_layers=int(config.hidden_layers),
    ).to(x)
    output_init = str(config.output_init).lower()
    if output_init == "zero":
        output_layer = model.net[-1]
        if not isinstance(output_layer, torch.nn.Linear):
            raise TypeError("ReLUFeedForward must end in a linear output layer")
        torch.nn.init.zeros_(output_layer.weight)
        torch.nn.init.zeros_(output_layer.bias)
    elif output_init != "default":
        raise ValueError("output_init must be 'default' or 'zero'")

    with torch.no_grad():
        initial_residuals = yc - model(xc)
        nuisance = torch.quantile(initial_residuals, levels, dim=0).T.contiguous()
        nuisance -= nuisance.mean(dim=1, keepdim=True)
    nuisance = torch.nn.Parameter(nuisance)
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "weight_decay": float(config.weight_decay)},
            {"params": [nuisance], "weight_decay": 0.0},
        ],
        lr=float(config.lr),
    )

    tau = levels.reshape(1, 1, -1)
    for step in range(1, int(config.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        residual = yc[:, :, None] - model(xc)[:, :, None] - nuisance[None, :, :]
        loss = torch.maximum(tau * residual, (tau - 1.0) * residual).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite ReLU CW-CQR loss at step {step}")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            nuisance -= nuisance.mean(dim=1, keepdim=True)

    # Re-express the location-equivalent fit as a conditional-mean predictor.
    with torch.no_grad():
        mean_shift = (yc - model(xc)).mean(dim=0, keepdim=True)
        y_mean = y_mean + mean_shift
    return ReLUFeedForwardFit(
        model_state={key: value.detach().clone() for key, value in model.state_dict().items()},
        input_dim=x.shape[1],
        output_dim=y.shape[1],
        width=int(config.width),
        hidden_layers=int(config.hidden_layers),
        x_mean=x_mean,
        y_mean=y_mean,
        method="relu_coordinatewise_cqr",
    )


def fit_relu_coordinatewise_asm(
    x: torch.Tensor,
    y: torch.Tensor,
    config: ReLUASMConfig | None = None,
) -> ReLUFeedForwardFit:
    """Fit ReLU ASM, using coordinatewise Huber ReLU as the default pilot."""

    config = config or ReLUASMConfig()
    pilot_cfg = config.pilot
    if pilot_cfg.width != config.width or pilot_cfg.hidden_layers != config.hidden_layers:
        pilot_cfg = ReLURegressionConfig(**{**pilot_cfg.__dict__, "width": config.width, "hidden_layers": config.hidden_layers})
    fit = fit_relu_coordinatewise_huber(x, y, pilot_cfg)
    for iteration in range(max(1, int(config.alt_iter))):
        with torch.no_grad():
            residuals = y - fit.predict(x)
        score_losses = [
            _IntegratedScoreLoss(
                estimate_antitonic_score(
                    residuals[:, j],
                    score_grid_size=config.score_grid_size,
                    kernel_points=config.kernel_points,
                    bandwidth=config.bandwidth,
                ),
                device=x.device,
                dtype=x.dtype,
            )
            for j in range(y.shape[1])
        ]

        def objective(residual: torch.Tensor) -> torch.Tensor:
            return torch.stack([score_losses[j](residual[:, j]) for j in range(y.shape[1])], dim=1)

        stage = ReLURegressionConfig(
            width=config.width,
            hidden_layers=config.hidden_layers,
            steps=config.steps,
            lr=config.lr,
            weight_decay=config.weight_decay,
            optimizer=config.optimizer,
            momentum=config.momentum,
            output_init=pilot_cfg.output_init,
            intercept_mode=pilot_cfg.intercept_mode,
            seed=config.seed + iteration,
            trace_every=config.trace_every,
        )
        fit = _fit_relu_objective(x, y, stage, objective, method="relu_coordinatewise_asm_huber_init", init_fit=fit)
    return fit
