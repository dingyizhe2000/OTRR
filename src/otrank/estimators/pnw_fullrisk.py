"""Full/profile empirical OT-rank regression with dense PNW OT scores."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from otrank.estimators.baselines import LinearModelFit, fit_zero_intercept_ols
from otrank.ot_maps.pnw import (
    PNWEntropicOTConfig,
    clear_pnw_sinkhorn_warm_start,
    median_cost_scale,
    normalize_ot_estimator_backend,
    pnw_scores,
    sample_pnw_reference,
)


def _load_torch(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys: list[str] = []
    existing_rows: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            existing_keys = list(reader.fieldnames or [])
            existing_rows = list(reader)
    keys = list(existing_keys)
    seen = set(keys)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    rewrite = bool(existing_keys) and keys != existing_keys
    mode = "w" if rewrite or not existing_keys else "a"
    with path.open(mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        if mode == "w":
            writer.writeheader()
            if rewrite:
                writer.writerows(existing_rows)
        writer.writerows(rows)
        fh.flush()


def _requested_stop() -> bool:
    stop_file = os.environ.get("OTRANK_STOP_FILE")
    return bool(stop_file) and Path(stop_file).exists()


@dataclass
class FullRiskPNWLinearConfig:
    """Hyperparameters for full/profile OT-rank linear regression."""

    pnw: PNWEntropicOTConfig = field(default_factory=PNWEntropicOTConfig)
    steps: int = 150
    lr: float = 1e-4
    l2_penalty: float = 0.0
    optimizer: str = "adam"
    eval_steps: tuple[int, ...] = (150,)
    trace_every: int = 50
    checkpoint_every: int = 50
    init_ridge: float = 1e-8
    intercept_mode: str = "zero"
    resume: bool = True
    checkpoint_scores: bool = False
    trace_first_step: bool = False
    trace_tensor_stats: bool = False


@dataclass
class FullRiskPNWLinearEstimator:
    """Fit the profile empirical OT-rank objective using dense PNW scores.

    The PNW score is recomputed from the current regression residuals inside
    every optimization step.
    """

    config: FullRiskPNWLinearConfig = field(default_factory=FullRiskPNWLinearConfig)
    init_fit_path: str | Path | None = None
    save_dir: str | Path | None = None

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> LinearModelFit:
        save_dir = Path(self.save_dir) if self.save_dir is not None else None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "config.json").write_text(
                json.dumps(
                    {
                        "estimator": "fullrisk_ot_rank_linear",
                        "steps": int(self.config.steps),
                        "lr": float(self.config.lr),
                        "l2_penalty": float(self.config.l2_penalty),
                        "optimizer": str(self.config.optimizer),
                        "eval_steps": list(self.config.eval_steps),
                        "trace_every": int(self.config.trace_every),
                        "checkpoint_every": int(self.config.checkpoint_every),
                        "checkpoint_scores": bool(self.config.checkpoint_scores),
                        "trace_first_step": bool(self.config.trace_first_step),
                        "trace_tensor_stats": bool(self.config.trace_tensor_stats),
                        "intercept_mode": str(self.config.intercept_mode),
                        "init_fit_path": str(self.init_fit_path) if self.init_fit_path is not None else None,
                        "pnw": {
                            "ot_estimator": str(self.config.pnw.ot_estimator),
                            "m_ref": int(self.config.pnw.m_ref),
                            "epsilon": float(self.config.pnw.epsilon),
                            "sinkhorn_iters": int(self.config.pnw.sinkhorn_iters),
                            "reference_distribution": str(self.config.pnw.reference_distribution),
                            "reference_radius": float(self.config.pnw.reference_radius),
                            "reference_values_explicit": self.config.pnw.reference_values is not None,
                            "seed": int(self.config.pnw.seed),
                            "cost_scale": self.config.pnw.cost_scale,
                            "topk_barycentric": self.config.pnw.topk_barycentric,
                            "chunk_size": self.config.pnw.chunk_size,
                            "detach_sinkhorn_gradient": self.config.pnw.detach_sinkhorn_gradient,
                            "backend": str(self.config.pnw.backend),
                            "implicit_solve_dtype": str(self.config.pnw.implicit_solve_dtype),
                            "implicit_gmres_iters": int(self.config.pnw.implicit_gmres_iters),
                            "implicit_gmres_tol": float(self.config.pnw.implicit_gmres_tol),
                            "implicit_ridge": float(self.config.pnw.implicit_ridge),
                            "implicit_chunk_size": int(self.config.pnw.implicit_chunk_size),
                            "implicit_cache_dtype": str(self.config.pnw.implicit_cache_dtype),
                            "sinkhorn_warm_start": bool(self.config.pnw.sinkhorn_warm_start),
                            "triton_direct_block_n": int(self.config.pnw.triton_direct_block_n),
                            "triton_direct_block_m": int(self.config.pnw.triton_direct_block_m),
                            "triton_matvec_block_n": int(self.config.pnw.triton_matvec_block_n),
                            "triton_matvec_block_m": int(self.config.pnw.triton_matvec_block_m),
                            "triton_implicit_add_block_m": int(self.config.pnw.triton_implicit_add_block_m),
                            "triton_num_warps": int(self.config.pnw.triton_num_warps),
                        },
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

        try:
            pilot = fit_zero_intercept_ols(x, y, ridge=float(self.config.init_ridge))
        except RuntimeError as exc:
            if x.dtype not in {torch.float16, torch.bfloat16}:
                raise
            # CUDA linear solves do not support half/bfloat16 on all backends.
            # The fallback pilot is only used when no explicit initializer is
            # available, so compute it in fp32 and cast back to the requested
            # training dtype.
            pilot32 = fit_zero_intercept_ols(x.float(), y.float(), ridge=float(self.config.init_ridge))
            pilot = LinearModelFit(
                coef=pilot32.coef.to(device=x.device, dtype=x.dtype),
                x_mean=pilot32.x_mean.to(device=x.device, dtype=x.dtype),
                y_mean=pilot32.y_mean.to(device=x.device, dtype=x.dtype),
                method=f"{pilot32.method}_fp32_fallback",
            )
        coef0, init_x_mean, init_y_mean = self._load_initial_fit(x.device, x.dtype, fallback=pilot)
        mode = self.config.intercept_mode.lower()
        if mode == "zero":
            x_mean = torch.zeros(1, x.shape[1], device=x.device, dtype=x.dtype)
            y_mean = torch.zeros(1, y.shape[1], device=y.device, dtype=y.dtype)
        elif mode == "pilot":
            x_mean = init_x_mean
            y_mean = init_y_mean
        else:
            raise ValueError("intercept_mode must be 'zero' or 'pilot'")
        coef = torch.nn.Parameter(coef0.detach().clone())
        target = sample_pnw_reference(
            n=int(self.config.pnw.m_ref),
            d=y.shape[1],
            config=self.config.pnw,
            device=x.device,
            dtype=x.dtype,
        )
        with torch.no_grad():
            init_residuals = y - ((x - x_mean) @ coef.detach().T + y_mean)
            cost_scale = (
                float(self.config.pnw.cost_scale)
                if self.config.pnw.cost_scale is not None
                else median_cost_scale(init_residuals, target)
            )

        optimizer_name = str(self.config.optimizer).lower()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam([coef], lr=float(self.config.lr))
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD([coef], lr=float(self.config.lr))
        else:
            raise ValueError("FullRiskPNWLinearConfig.optimizer must be 'adam' or 'sgd'")
        l2_penalty = float(self.config.l2_penalty)
        if l2_penalty < 0.0:
            raise ValueError("l2_penalty must be nonnegative")
        start_step = 0
        if save_dir is not None and self.config.resume:
            loaded = self._load_checkpoint(save_dir, coef, optimizer)
            start_step = int(loaded)

        eval_steps = {int(step) for step in self.config.eval_steps if 1 <= int(step) <= int(self.config.steps)}
        if not eval_steps:
            eval_steps = {int(self.config.steps)}
        trace_every = max(1, int(self.config.trace_every))
        checkpoint_every = max(1, int(self.config.checkpoint_every))
        trace_rows: list[dict[str, Any]] = []
        start_time = time.perf_counter()
        last_trace_time = start_time
        last_trace_step = int(start_step)
        warm_start_id = id(coef) if bool(self.config.pnw.sinkhorn_warm_start) else 0
        for step in range(start_step + 1, int(self.config.steps) + 1):
            if _requested_stop():
                if save_dir is not None:
                    self._save_checkpoint(save_dir, max(start_step, step - 1), coef, optimizer, x_mean, y_mean)
                if warm_start_id:
                    clear_pnw_sinkhorn_warm_start(int(warm_start_id))
                raise InterruptedError(f"full-risk PNW fit preempted before step {step}")
            optimizer.zero_grad(set_to_none=True)
            residuals = y - ((x - x_mean) @ coef.T + y_mean)
            if bool(self.config.checkpoint_scores):
                def _loss_scores(residual_arg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    loss_value, score_value, _ = pnw_scores(
                        residual_arg,
                        target,
                        ot_estimator=str(self.config.pnw.ot_estimator),
                        epsilon=float(self.config.pnw.epsilon),
                        sinkhorn_iters=int(self.config.pnw.sinkhorn_iters),
                        cost_scale=cost_scale,
                        topk_barycentric=self.config.pnw.topk_barycentric,
                        chunk_size=self.config.pnw.chunk_size,
                        detach_sinkhorn_gradient=self.config.pnw.detach_sinkhorn_gradient,
                        backend=str(self.config.pnw.backend),
                        implicit_solve_dtype=str(self.config.pnw.implicit_solve_dtype),
                        implicit_gmres_iters=int(self.config.pnw.implicit_gmres_iters),
                        implicit_gmres_tol=float(self.config.pnw.implicit_gmres_tol),
                        implicit_ridge=float(self.config.pnw.implicit_ridge),
                        implicit_chunk_size=int(self.config.pnw.implicit_chunk_size),
                        implicit_cache_dtype=str(self.config.pnw.implicit_cache_dtype),
                        sinkhorn_warm_start_id=int(warm_start_id),
                        triton_direct_block_n=int(self.config.pnw.triton_direct_block_n),
                        triton_direct_block_m=int(self.config.pnw.triton_direct_block_m),
                        triton_matvec_block_n=int(self.config.pnw.triton_matvec_block_n),
                        triton_matvec_block_m=int(self.config.pnw.triton_matvec_block_m),
                        triton_implicit_add_block_m=int(self.config.pnw.triton_implicit_add_block_m),
                        triton_num_warps=int(self.config.pnw.triton_num_warps),
                    )
                    return loss_value, score_value

                loss, scores = torch.utils.checkpoint.checkpoint(_loss_scores, residuals, use_reentrant=False)
            else:
                loss, scores, _ = pnw_scores(
                    residuals,
                    target,
                    ot_estimator=str(self.config.pnw.ot_estimator),
                    epsilon=float(self.config.pnw.epsilon),
                    sinkhorn_iters=int(self.config.pnw.sinkhorn_iters),
                    cost_scale=cost_scale,
                    topk_barycentric=self.config.pnw.topk_barycentric,
                    chunk_size=self.config.pnw.chunk_size,
                    detach_sinkhorn_gradient=self.config.pnw.detach_sinkhorn_gradient,
                    backend=str(self.config.pnw.backend),
                    implicit_solve_dtype=str(self.config.pnw.implicit_solve_dtype),
                    implicit_gmres_iters=int(self.config.pnw.implicit_gmres_iters),
                    implicit_gmres_tol=float(self.config.pnw.implicit_gmres_tol),
                    implicit_ridge=float(self.config.pnw.implicit_ridge),
                    implicit_chunk_size=int(self.config.pnw.implicit_chunk_size),
                    implicit_cache_dtype=str(self.config.pnw.implicit_cache_dtype),
                    sinkhorn_warm_start_id=int(warm_start_id),
                    triton_direct_block_n=int(self.config.pnw.triton_direct_block_n),
                    triton_direct_block_m=int(self.config.pnw.triton_direct_block_m),
                    triton_matvec_block_n=int(self.config.pnw.triton_matvec_block_n),
                    triton_matvec_block_m=int(self.config.pnw.triton_matvec_block_m),
                    triton_implicit_add_block_m=int(self.config.pnw.triton_implicit_add_block_m),
                    triton_num_warps=int(self.config.pnw.triton_num_warps),
                )
            ot_loss = loss
            penalty_loss = 0.5 * l2_penalty * torch.sum(coef.square())
            loss = ot_loss + penalty_loss
            if not torch.isfinite(loss):
                if warm_start_id:
                    clear_pnw_sinkhorn_warm_start(int(warm_start_id))
                raise RuntimeError(f"non-finite full-risk PNW loss at step {step}")
            loss.backward()
            optimizer.step()

            should_trace = (
                (bool(self.config.trace_first_step) and step == 1)
                or step % trace_every == 0
                or step in eval_steps
                or step == int(self.config.steps)
            )
            if should_trace:
                now = time.perf_counter()
                step_delta = max(1, int(step) - int(last_trace_step))
                time_delta = max(1e-12, now - last_trace_time)
                trace_row = {
                    "step": step,
                    "elapsed_sec": float(now - start_time),
                    "sec_per_step": float(time_delta / step_delta),
                    "steps_per_sec": float(step_delta / time_delta),
                    "cost_scale": float(cost_scale),
                }
                if bool(self.config.trace_tensor_stats):
                    trace_row.update(
                        {
                            "loss": float(loss.detach().item()),
                            "ot_loss": float(ot_loss.detach().item()),
                            "l2_penalty_loss": float(penalty_loss.detach().item()),
                            "coef_delta_from_init": float(torch.linalg.norm(coef.detach() - coef0, ord="fro").item()),
                            "grad_norm": (
                                float(torch.linalg.norm(coef.grad.detach()).item())
                                if coef.grad is not None
                                else float("nan")
                            ),
                            "score_norm_mean": (
                                float(torch.linalg.norm(scores.detach(), dim=1).mean().item())
                                if scores.numel() > 0
                                else float("nan")
                            ),
                        }
                    )
                trace_rows.append(trace_row)
                last_trace_time = now
                last_trace_step = int(step)
                if save_dir is not None:
                    _append_csv(save_dir / "trace.csv", trace_rows)
                    trace_rows = []

            if save_dir is not None and (step % checkpoint_every == 0 or step in eval_steps or step == int(self.config.steps)):
                self._save_checkpoint(save_dir, step, coef, optimizer, x_mean, y_mean)

            if save_dir is not None and step in eval_steps:
                self._save_iteration(save_dir, step, coef, x_mean, y_mean)

            if _requested_stop():
                if save_dir is not None:
                    self._save_checkpoint(save_dir, step, coef, optimizer, x_mean, y_mean)
                if warm_start_id:
                    clear_pnw_sinkhorn_warm_start(int(warm_start_id))
                raise InterruptedError(f"full-risk PNW fit preempted after step {step}")

        if save_dir is not None and trace_rows:
            _append_csv(save_dir / "trace.csv", trace_rows)
        if warm_start_id:
            clear_pnw_sinkhorn_warm_start(int(warm_start_id))
        estimator_name, _ = self._ot_estimator_backend()
        return LinearModelFit(coef=coef.detach(), x_mean=x_mean, y_mean=y_mean, method=f"ot_rank_{estimator_name}_fullrisk")

    def _ot_estimator_backend(self) -> tuple[str, str]:
        return normalize_ot_estimator_backend(str(self.config.pnw.ot_estimator), str(self.config.pnw.backend))

    def _load_initial_fit(
        self,
        device: torch.device,
        dtype: torch.dtype,
        *,
        fallback: LinearModelFit,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.init_fit_path is None:
            return fallback.coef.detach().clone(), fallback.x_mean.detach().clone(), fallback.y_mean.detach().clone()
        path = Path(self.init_fit_path)
        if not path.exists():
            return fallback.coef.detach().clone(), fallback.x_mean.detach().clone(), fallback.y_mean.detach().clone()
        state = _load_torch(path, device)
        coef = state.get("coef")
        if coef is None:
            return fallback.coef.detach().clone(), fallback.x_mean.detach().clone(), fallback.y_mean.detach().clone()
        coef = coef.to(device=device, dtype=dtype)
        if tuple(coef.shape) != tuple(fallback.coef.shape):
            return fallback.coef.detach().clone(), fallback.x_mean.detach().clone(), fallback.y_mean.detach().clone()
        x_mean = state.get("x_mean")
        y_mean = state.get("y_mean")
        if x_mean is None:
            x_mean = torch.zeros_like(fallback.x_mean)
        if y_mean is None:
            y_mean = torch.zeros_like(fallback.y_mean)
        return (
            coef.detach().clone(),
            x_mean.to(device=device, dtype=dtype).detach().clone(),
            y_mean.to(device=device, dtype=dtype).detach().clone(),
        )

    def _load_checkpoint(self, save_dir: Path, coef: torch.nn.Parameter, optimizer: torch.optim.Optimizer) -> int:
        path = save_dir / "checkpoint.pt"
        if not path.exists():
            return 0
        state = _load_torch(path, coef.device)
        coef_state = state.get("coef")
        if coef_state is None:
            return 0
        with torch.no_grad():
            coef.copy_(coef_state.to(device=coef.device, dtype=coef.dtype))
        opt_state = state.get("optimizer")
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
        return int(state.get("step", 0))

    def _save_checkpoint(
        self,
        save_dir: Path,
        step: int,
        coef: torch.nn.Parameter,
        optimizer: torch.optim.Optimizer,
        x_mean: torch.Tensor,
        y_mean: torch.Tensor,
    ) -> None:
        torch.save(
            {
                "step": int(step),
                "coef": coef.detach().cpu(),
                "x_mean": x_mean.detach().cpu(),
                "y_mean": y_mean.detach().cpu(),
                "optimizer": optimizer.state_dict(),
            },
            save_dir / "checkpoint.pt",
        )

    def _save_iteration(
        self,
        save_dir: Path,
        step: int,
        coef: torch.nn.Parameter,
        x_mean: torch.Tensor,
        y_mean: torch.Tensor,
    ) -> None:
        iter_dir = save_dir / f"iteration_{int(step):03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "iteration": int(step),
                "coef": coef.detach().cpu(),
                "x_mean": x_mean.detach().cpu(),
                "y_mean": y_mean.detach().cpu(),
                "method": "ot_rank_pnw_fullrisk",
            },
            iter_dir / "state.pt",
        )
