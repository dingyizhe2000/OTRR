"""Public Python interface for OT-rank regression."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from .estimators import (
    ASMConfig,
    FullRiskPNWLinearConfig,
    FullRiskPNWLinearEstimator,
    FullRiskPNWReLUConfig,
    FullRiskPNWReLUEstimator,
    LinearModelFit,
    MCQROTConfig,
    ReLUCQRConfig,
    ReLUFeedForwardFit,
    fit_coordinatewise_asm,
    fit_coordinatewise_huber,
    fit_mcqr_ot_lens,
    fit_radial_huber,
    fit_relu_coordinatewise_cqr,
    fit_zero_intercept_ols,
)
from .inference import ASMInferenceResult, coordinatewise_asm_inference
from .ot_maps import PNWEntropicOTConfig

BaselineName = Literal["zero_intercept_ols", "huber", "radial_huber", "asm", "mcqr"]


@dataclass
class LinearOTRankConfig:
    """Default full/profile OT-rank settings."""

    ot_estimator: str = "pnw"
    m_ref: int = 131072
    epsilon: float = 0.10
    sinkhorn_iters: int = 1
    steps: int = 100
    lr: float = 0.0001
    trace_every: int = 50
    checkpoint_every: int = 50
    trace_first_step: bool = False
    trace_tensor_stats: bool = False
    reference_seed: int = 20260721
    backend: str = "dense_triton"
    implicit_gmres_iters: int = 30
    implicit_gmres_tol: float = 1e-6
    implicit_cache_dtype: str = "float32"
    sinkhorn_warm_start: bool = True
    triton_direct_block_n: int = 8
    triton_direct_block_m: int = 1024
    triton_matvec_block_n: int = 32
    triton_matvec_block_m: int = 512
    triton_implicit_add_block_m: int = 512
    triton_num_warps: int = 4
    dtype: str = "float32"
    device: str = "cpu"
    intercept_mode: Literal["zero", "pilot"] = "zero"

    def make_fullrisk_config(self) -> FullRiskPNWLinearConfig:
        return FullRiskPNWLinearConfig(
            pnw=PNWEntropicOTConfig(
                ot_estimator=str(self.ot_estimator),
                m_ref=int(self.m_ref),
                epsilon=float(self.epsilon),
                sinkhorn_iters=int(self.sinkhorn_iters),
                seed=int(self.reference_seed),
                backend=str(self.backend),
                implicit_gmres_iters=int(self.implicit_gmres_iters),
                implicit_gmres_tol=float(self.implicit_gmres_tol),
                implicit_cache_dtype=str(self.implicit_cache_dtype),
                sinkhorn_warm_start=bool(self.sinkhorn_warm_start),
                triton_direct_block_n=int(self.triton_direct_block_n),
                triton_direct_block_m=int(self.triton_direct_block_m),
                triton_matvec_block_n=int(self.triton_matvec_block_n),
                triton_matvec_block_m=int(self.triton_matvec_block_m),
                triton_implicit_add_block_m=int(self.triton_implicit_add_block_m),
                triton_num_warps=int(self.triton_num_warps),
            ),
            steps=int(self.steps),
            lr=float(self.lr),
            eval_steps=(int(self.steps),),
            trace_every=int(self.trace_every),
            checkpoint_every=int(self.checkpoint_every),
            trace_first_step=bool(self.trace_first_step),
            trace_tensor_stats=bool(self.trace_tensor_stats),
            intercept_mode=str(self.intercept_mode),
        )


@dataclass
class ReLUOTRankConfig:
    """Convenience settings for a CW-CQR-initialized ReLU OTRR fit."""

    width: int = 16
    hidden_layers: int = 3
    pilot_steps: int = 1000
    pilot_lr: float = 1e-3
    pilot_weight_decay: float = 1e-3
    pilot_seed: int = 20260721
    m_ref: int = 131072
    epsilon: float = 0.10
    sinkhorn_iters: int = 1
    steps: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-3
    reference_seed: int = 20260721
    backend: str = "dense_triton"
    implicit_gmres_iters: int = 30
    implicit_gmres_tol: float = 1e-6
    intercept_mode: Literal["zero", "center"] = "center"

    def make_pilot_config(self) -> ReLUCQRConfig:
        return ReLUCQRConfig(
            width=int(self.width),
            hidden_layers=int(self.hidden_layers),
            steps=int(self.pilot_steps),
            lr=float(self.pilot_lr),
            weight_decay=float(self.pilot_weight_decay),
            intercept_mode=str(self.intercept_mode),
            seed=int(self.pilot_seed),
        )

    def make_fullrisk_config(self) -> FullRiskPNWReLUConfig:
        return FullRiskPNWReLUConfig(
            pnw=PNWEntropicOTConfig(
                m_ref=int(self.m_ref),
                epsilon=float(self.epsilon),
                sinkhorn_iters=int(self.sinkhorn_iters),
                seed=int(self.reference_seed),
                backend=str(self.backend),
                implicit_gmres_iters=int(self.implicit_gmres_iters),
                implicit_gmres_tol=float(self.implicit_gmres_tol),
            ),
            width=int(self.width),
            hidden_layers=int(self.hidden_layers),
            steps=int(self.steps),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
            eval_steps=(int(self.steps),),
            trace_every=max(1, int(self.steps)),
            checkpoint_every=max(1, int(self.steps)),
            intercept_mode=str(self.intercept_mode),
            seed=int(self.reference_seed),
        )


def as_tensor_pair(x, y, *, device: str = "cpu", dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert array-like ``x`` and ``y`` to a same-device tensor pair."""

    return torch.as_tensor(x, device=device, dtype=dtype), torch.as_tensor(y, device=device, dtype=dtype)


def fit_linear_baseline(
    x: torch.Tensor,
    y: torch.Tensor,
    method: BaselineName = "zero_intercept_ols",
    *,
    rscript: str = "Rscript",
) -> LinearModelFit:
    """Fit one of the linear baselines used in the paper."""

    if method == "zero_intercept_ols":
        return fit_zero_intercept_ols(x, y, ridge=1e-8)
    if method == "huber":
        return fit_coordinatewise_huber(x, y, tau=1.345, max_iter=80, ridge=1e-8, intercept_mode="zero")
    if method == "radial_huber":
        return fit_radial_huber(x, y, tau_multiplier=1.345, max_iter=80, ridge=1e-8, intercept_mode="zero")
    if method == "asm":
        return fit_coordinatewise_asm(
            x,
            y,
            ASMConfig(
                betapilot="OLS",
                symmetric=False,
                alt_iter=1,
                max_iter=65,
                intercept_selection="mean",
                center_xy=False,
                force_zero_intercept=True,
                rscript=rscript,
            ),
        )
    if method == "mcqr":
        return fit_mcqr_ot_lens(
            x,
            y,
            MCQROTConfig(max_iter=200, lr=0.02, ridge=1e-5, reference="normal", seed=2026, intercept_mode="zero"),
        )
    raise ValueError(f"Unknown baseline method: {method!r}")


def fit_linear_asm_inference(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    config: ASMConfig | None = None,
    rscript: str = "Rscript",
    center_x: bool = True,
) -> ASMInferenceResult:
    """Fit CW-ASM once and return package and joint-score inference outputs."""

    asm_config = config or ASMConfig(
        betapilot="OLS",
        symmetric=False,
        alt_iter=1,
        max_iter=65,
        intercept_selection="mean",
        center_xy=False,
        force_zero_intercept=True,
        rscript=rscript,
    )
    return coordinatewise_asm_inference(
        x,
        y,
        asm_config,
        center_x=center_x,
    )


def fit_ot_rank_linear(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    init_fit: LinearModelFit | None = None,
    init_method: BaselineName = "zero_intercept_ols",
    config: LinearOTRankConfig | None = None,
    save_dir: str | Path | None = None,
    rscript: str = "Rscript",
) -> LinearModelFit:
    """Fit full/profile OT-rank from a pilot linear estimator."""

    import tempfile

    cfg = config or LinearOTRankConfig()
    pilot = init_fit or fit_linear_baseline(x, y, init_method, rscript=rscript)
    with tempfile.TemporaryDirectory(prefix="otrank_init_") as tmp:
        init_path = Path(tmp) / "init.pt"
        torch.save(
            {
                "coef": pilot.coef.detach().cpu(),
                "x_mean": pilot.x_mean.detach().cpu(),
                "y_mean": pilot.y_mean.detach().cpu(),
                "method": pilot.method,
            },
            init_path,
        )
        return FullRiskPNWLinearEstimator(
            config=cfg.make_fullrisk_config(),
            init_fit_path=init_path,
            save_dir=save_dir,
        ).fit(x, y)


def fit_ot_rank_relu(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    init_fit: ReLUFeedForwardFit | None = None,
    config: ReLUOTRankConfig | None = None,
    save_dir: str | Path | None = None,
) -> ReLUFeedForwardFit:
    """Fit one CW-CQR-initialized full/profile ReLU OTRR member."""

    import tempfile

    cfg = config or ReLUOTRankConfig()
    pilot = init_fit or fit_relu_coordinatewise_cqr(x, y, cfg.make_pilot_config())
    with tempfile.TemporaryDirectory(prefix="otrank_relu_init_") as tmp:
        init_path = Path(tmp) / "init.pt"
        torch.save(pilot.state_dict_for_save(), init_path)
        return FullRiskPNWReLUEstimator(
            config=cfg.make_fullrisk_config(),
            init_fit_path=init_path,
            save_dir=save_dir,
        ).fit(x, y)


__all__ = [
    "LinearOTRankConfig",
    "ReLUOTRankConfig",
    "as_tensor_pair",
    "fit_linear_baseline",
    "fit_ot_rank_linear",
    "fit_ot_rank_relu",
]
