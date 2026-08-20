"""Full/profile PNW regression over small ReLU feedforward networks."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

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
                keys.append(key)
                seen.add(key)
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


class ReLUFeedForward(torch.nn.Module):
    """Small fully connected ReLU network mapping covariates to responses."""

    def __init__(self, input_dim: int, output_dim: int, *, width: int = 4, hidden_layers: int = 5) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if width <= 0:
            raise ValueError("width must be positive")
        if hidden_layers < 0:
            raise ValueError("hidden_layers must be nonnegative")
        layers: list[torch.nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(hidden_layers)):
            layers.append(torch.nn.Linear(in_dim, int(width)))
            layers.append(torch.nn.ReLU())
            in_dim = int(width)
        layers.append(torch.nn.Linear(in_dim, int(output_dim)))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ReLUFeedForwardFit:
    """Serializable fitted ReLU regression model."""

    model_state: dict[str, torch.Tensor]
    input_dim: int
    output_dim: int
    width: int
    hidden_layers: int
    x_mean: torch.Tensor
    y_mean: torch.Tensor
    method: str

    def _model(self, *, device: torch.device, dtype: torch.dtype) -> ReLUFeedForward:
        model = ReLUFeedForward(
            int(self.input_dim),
            int(self.output_dim),
            width=int(self.width),
            hidden_layers=int(self.hidden_layers),
        ).to(device=device, dtype=dtype)
        state = {key: value.to(device=device, dtype=dtype) for key, value in self.model_state.items()}
        model.load_state_dict(state)
        model.eval()
        return model

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        model = self._model(device=x.device, dtype=x.dtype)
        x_mean = self.x_mean.to(device=x.device, dtype=x.dtype)
        y_mean = self.y_mean.to(device=x.device, dtype=x.dtype)
        with torch.no_grad():
            return model(x - x_mean) + y_mean

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-observation output Jacobians with shape ``(n, d, p)``."""

        model = self._model(device=x.device, dtype=x.dtype)
        x_mean = self.x_mean.to(device=x.device, dtype=x.dtype)
        inputs = x.detach().clone().requires_grad_(True)
        outputs = model(inputs - x_mean)
        gradients = [
            torch.autograd.grad(outputs[:, j].sum(), inputs, retain_graph=j + 1 < outputs.shape[1])[0]
            for j in range(outputs.shape[1])
        ]
        return torch.stack(gradients, dim=1).detach()

    def state_dict_for_save(self) -> dict[str, Any]:
        return {
            "model_type": "relu_feedforward",
            "model_state": {key: value.detach().cpu() for key, value in self.model_state.items()},
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
            "width": int(self.width),
            "hidden_layers": int(self.hidden_layers),
            "x_mean": self.x_mean.detach().cpu(),
            "y_mean": self.y_mean.detach().cpu(),
            "method": str(self.method),
        }


def relu_fit_from_state(state: dict[str, Any], *, device: torch.device, dtype: torch.dtype | None = None) -> ReLUFeedForwardFit:
    if str(state.get("model_type", "")) != "relu_feedforward":
        raise ValueError("state is not a ReLU feedforward fit")
    target_dtype = dtype or torch.float32
    model_state = {
        str(key): value.to(device=device, dtype=target_dtype).detach().clone()
        for key, value in dict(state["model_state"]).items()
    }
    x_mean = state.get("x_mean")
    y_mean = state.get("y_mean")
    if x_mean is None or y_mean is None:
        raise KeyError("ReLU state must contain x_mean and y_mean")
    return ReLUFeedForwardFit(
        model_state=model_state,
        input_dim=int(state["input_dim"]),
        output_dim=int(state["output_dim"]),
        width=int(state["width"]),
        hidden_layers=int(state["hidden_layers"]),
        x_mean=x_mean.to(device=device, dtype=target_dtype),
        y_mean=y_mean.to(device=device, dtype=target_dtype),
        method=str(state.get("method", "relu_pnw_fullrisk")),
    )


@dataclass
class FullRiskPNWReLUConfig:
    """Hyperparameters for profile PNW regression over ReLU networks."""

    pnw: PNWEntropicOTConfig = field(default_factory=PNWEntropicOTConfig)
    width: int = 16
    hidden_layers: int = 3
    steps: int = 150
    lr: float = 1e-4
    weight_decay: float = 1e-3
    optimizer: str = "adam"
    eval_steps: tuple[int, ...] = (150,)
    trace_every: int = 50
    checkpoint_every: int = 50
    intercept_mode: str = "zero"
    seed: int = 20260721
    resume: bool = True
    init_mse_steps: int = 0
    init_mse_lr: float = 1e-3
    trace_first_step: bool = False
    trace_tensor_stats: bool = False


@dataclass
class FullRiskPNWReLUEstimator:
    """Fit full/profile PNW risk with a ReLU feedforward candidate class."""

    config: FullRiskPNWReLUConfig = field(default_factory=FullRiskPNWReLUConfig)
    save_dir: str | Path | None = None
    init_fit_path: str | Path | None = None

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> ReLUFeedForwardFit:
        save_dir = Path(self.save_dir) if self.save_dir is not None else None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "config.json").write_text(
                json.dumps(self._config_for_json(), indent=2, sort_keys=True),
                encoding="utf-8",
            )

        mode = str(self.config.intercept_mode).lower()
        if mode == "zero":
            x_mean = torch.zeros(1, x.shape[1], device=x.device, dtype=x.dtype)
            y_mean = torch.zeros(1, y.shape[1], device=y.device, dtype=y.dtype)
        elif mode in {"center", "intercept"}:
            x_mean = x.mean(dim=0, keepdim=True)
            y_mean = y.mean(dim=0, keepdim=True)
        else:
            raise ValueError("intercept_mode must be 'zero', 'center', or 'intercept'")

        self._ot_estimator_backend()
        model = self._init_model(x, y)
        if self.init_fit_path is not None:
            init_path = Path(self.init_fit_path)
            if not init_path.exists():
                raise FileNotFoundError(f"ReLU PNW init fit does not exist: {init_path}")
            init_state = _load_torch(init_path, x.device)
            init_fit = relu_fit_from_state(init_state, device=x.device, dtype=x.dtype)
            if (
                init_fit.input_dim != x.shape[1]
                or init_fit.output_dim != y.shape[1]
                or init_fit.width != int(self.config.width)
                or init_fit.hidden_layers != int(self.config.hidden_layers)
            ):
                raise ValueError("ReLU PNW init architecture does not match the requested model")
            model.load_state_dict(init_fit.model_state)
            x_mean = init_fit.x_mean.to(device=x.device, dtype=x.dtype)
            y_mean = init_fit.y_mean.to(device=y.device, dtype=y.dtype)
        optimizer_name = str(self.config.optimizer).lower()
        if optimizer_name != "adam":
            raise ValueError("Full-risk PNW ReLU implementation currently uses Adam only")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay),
        )

        if (
            self.init_fit_path is None
            and int(self.config.init_mse_steps) > 0
            and not (save_dir is not None and (save_dir / "checkpoint.pt").exists())
        ):
            self._mse_warm_start(model, x, y, x_mean, y_mean)

        start_step = 0
        if save_dir is not None and bool(self.config.resume):
            start_step = self._load_checkpoint(save_dir, model, optimizer)

        target = sample_pnw_reference(
            n=int(self.config.pnw.m_ref),
            d=y.shape[1],
            config=self.config.pnw,
            device=x.device,
            dtype=x.dtype,
        )
        with torch.no_grad():
            init_residuals = y - (model(x - x_mean) + y_mean)
            cost_scale = (
                float(self.config.pnw.cost_scale)
                if self.config.pnw.cost_scale is not None
                else median_cost_scale(init_residuals, target)
            )

        eval_steps = {int(step) for step in self.config.eval_steps if 1 <= int(step) <= int(self.config.steps)}
        if not eval_steps:
            eval_steps = {int(self.config.steps)}
        trace_every = max(1, int(self.config.trace_every))
        checkpoint_every = max(1, int(self.config.checkpoint_every))
        trace_rows: list[dict[str, Any]] = []
        start_time = time.perf_counter()
        last_trace_time = start_time
        last_trace_step = int(start_step)
        warm_start_id = id(model) if bool(self.config.pnw.sinkhorn_warm_start) else 0

        for step in range(start_step + 1, int(self.config.steps) + 1):
            if _requested_stop():
                if save_dir is not None:
                    self._save_checkpoint(save_dir, max(start_step, step - 1), model, optimizer, x_mean, y_mean)
                if warm_start_id:
                    clear_pnw_sinkhorn_warm_start(int(warm_start_id))
                raise InterruptedError(f"full-risk PNW ReLU fit preempted before step {step}")

            optimizer.zero_grad(set_to_none=True)
            residuals = y - (model(x - x_mean) + y_mean)
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
            if not torch.isfinite(loss):
                if warm_start_id:
                    clear_pnw_sinkhorn_warm_start(int(warm_start_id))
                raise RuntimeError(f"non-finite full-risk PNW ReLU loss at step {step}")
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
                    "step": int(step),
                    "elapsed_sec": float(now - start_time),
                    "sec_per_step": float(time_delta / step_delta),
                    "steps_per_sec": float(step_delta / time_delta),
                    "cost_scale": float(cost_scale),
                }
                if bool(self.config.trace_tensor_stats):
                    grad_sq = 0.0
                    for param in model.parameters():
                        if param.grad is not None:
                            grad_sq += float(torch.sum(param.grad.detach().float().pow(2)).item())
                    trace_row.update(
                        {
                            "loss": float(loss.detach().item()),
                            "grad_norm": grad_sq**0.5,
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
                self._save_checkpoint(save_dir, step, model, optimizer, x_mean, y_mean)

            if save_dir is not None and step in eval_steps:
                self._save_iteration(save_dir, step, model, x_mean, y_mean)

            if _requested_stop():
                if save_dir is not None:
                    self._save_checkpoint(save_dir, step, model, optimizer, x_mean, y_mean)
                if warm_start_id:
                    clear_pnw_sinkhorn_warm_start(int(warm_start_id))
                raise InterruptedError(f"full-risk PNW ReLU fit preempted after step {step}")

        if save_dir is not None and trace_rows:
            _append_csv(save_dir / "trace.csv", trace_rows)
        if warm_start_id:
            clear_pnw_sinkhorn_warm_start(int(warm_start_id))
        return self._fit_from_model(model, x_mean, y_mean, method="relu_pnw_fullrisk")

    def _config_for_json(self) -> dict[str, Any]:
        return {
            "estimator": "fullrisk_pnw_relu",
            "width": int(self.config.width),
            "hidden_layers": int(self.config.hidden_layers),
            "steps": int(self.config.steps),
            "lr": float(self.config.lr),
            "weight_decay": float(self.config.weight_decay),
            "optimizer": str(self.config.optimizer),
            "eval_steps": list(self.config.eval_steps),
            "trace_every": int(self.config.trace_every),
            "checkpoint_every": int(self.config.checkpoint_every),
            "intercept_mode": str(self.config.intercept_mode),
            "seed": int(self.config.seed),
            "init_mse_steps": int(self.config.init_mse_steps),
            "init_mse_lr": float(self.config.init_mse_lr),
            "init_fit_path": None if self.init_fit_path is None else str(self.init_fit_path),
            "pnw": {
                "ot_estimator": str(self.config.pnw.ot_estimator),
                "m_ref": int(self.config.pnw.m_ref),
                "epsilon": float(self.config.pnw.epsilon),
                "sinkhorn_iters": int(self.config.pnw.sinkhorn_iters),
                "reference_distribution": str(self.config.pnw.reference_distribution),
                "reference_radius": float(self.config.pnw.reference_radius),
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
        }

    def _ot_estimator_backend(self) -> tuple[str, str]:
        return normalize_ot_estimator_backend(str(self.config.pnw.ot_estimator), str(self.config.pnw.backend))

    def _init_model(self, x: torch.Tensor, y: torch.Tensor) -> ReLUFeedForward:
        cuda_devices = [int(x.device.index or 0)] if x.device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(self.config.seed))
            if x.device.type == "cuda":
                torch.cuda.manual_seed_all(int(self.config.seed))
            model = ReLUFeedForward(
                int(x.shape[1]),
                int(y.shape[1]),
                width=int(self.config.width),
                hidden_layers=int(self.config.hidden_layers),
            )
        return model.to(device=x.device, dtype=x.dtype)

    def _mse_warm_start(
        self,
        model: ReLUFeedForward,
        x: torch.Tensor,
        y: torch.Tensor,
        x_mean: torch.Tensor,
        y_mean: torch.Tensor,
    ) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.config.init_mse_lr))
        for _ in range(int(self.config.init_mse_steps)):
            optimizer.zero_grad(set_to_none=True)
            residuals = y - (model(x - x_mean) + y_mean)
            loss = torch.mean(torch.sum(residuals.pow(2), dim=1))
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite ReLU MSE warm-start loss")
            loss.backward()
            optimizer.step()

    def _load_checkpoint(self, save_dir: Path, model: ReLUFeedForward, optimizer: torch.optim.Optimizer) -> int:
        path = save_dir / "checkpoint.pt"
        if not path.exists():
            return 0
        state = _load_torch(path, next(model.parameters()).device)
        if str(state.get("model_type", "")) != "relu_feedforward":
            return 0
        model_state = state.get("model_state")
        if model_state is None:
            return 0
        model.load_state_dict(model_state)
        opt_state = state.get("optimizer")
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
        return int(state.get("step", 0))

    def _save_checkpoint(
        self,
        save_dir: Path,
        step: int,
        model: ReLUFeedForward,
        optimizer: torch.optim.Optimizer,
        x_mean: torch.Tensor,
        y_mean: torch.Tensor,
    ) -> None:
        state = self._fit_from_model(model, x_mean, y_mean, method="relu_pnw_fullrisk").state_dict_for_save()
        state.update({"step": int(step), "optimizer": optimizer.state_dict()})
        torch.save(state, save_dir / "checkpoint.pt")

    def _save_iteration(
        self,
        save_dir: Path,
        step: int,
        model: ReLUFeedForward,
        x_mean: torch.Tensor,
        y_mean: torch.Tensor,
    ) -> None:
        iter_dir = save_dir / f"iteration_{int(step):03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        state = self._fit_from_model(model, x_mean, y_mean, method="relu_pnw_fullrisk").state_dict_for_save()
        state["iteration"] = int(step)
        torch.save(state, iter_dir / "state.pt")

    def _fit_from_model(
        self,
        model: ReLUFeedForward,
        x_mean: torch.Tensor,
        y_mean: torch.Tensor,
        *,
        method: str,
    ) -> ReLUFeedForwardFit:
        return ReLUFeedForwardFit(
            model_state={key: value.detach().cpu() for key, value in model.state_dict().items()},
            input_dim=int(model.net[0].in_features) if int(self.config.hidden_layers) > 0 else int(model.net[0].in_features),
            output_dim=int(y_mean.shape[1]),
            width=int(self.config.width),
            hidden_layers=int(self.config.hidden_layers),
            x_mean=x_mean.detach().cpu(),
            y_mean=y_mean.detach().cpu(),
            method=method,
        )
