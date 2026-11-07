"""Simulation data generation, estimator fitting, and one-job evaluation."""

from __future__ import annotations

import json
import multiprocessing as mp
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_NUMERIC_THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "MALLOC_ARENA_MAX": "2",
}
for _env_name, _env_value in _NUMERIC_THREAD_ENV_DEFAULTS.items():
    os.environ.setdefault(_env_name, _env_value)

import torch
from tqdm import tqdm

from otrank.data.noise import make_generator, random_orthogonal
from otrank.data.sim import (
    gaussian_nonlinear_jacobian,
    make_linear_simulation,
    make_nonlinear_simulation,
)
from otrank.estimators import fit_coordinatewise_huber, fit_radial_huber, fit_zero_intercept_ols
from otrank.metrics import (
    coefficient_error,
    centered_prediction_mse,
    frobenius_angle,
    jacobian_relative_gap,
    jacobian_rmse_per_entry,
    jacobian_rowspace_diagnostics,
    mean_bias,
    prediction_mae,
    prediction_mse,
    prediction_mse_per_dim,
    prediction_rmse_per_dim,
    prediction_space_angle,
    principal_angles,
    rotation_prediction_gap,
    rotation_prediction_mse_per_dim,
    rotation_prediction_rmse_per_dim,
    translation_adjusted_mse,
    weighted_sine_discrepancy,
)

_PROGRESS_FILE = None

from .layout import (
    _find_init_fit_path, _is_iteration_method, _load_checkpoint,
    _load_completed_job_rows, _method_label, _noise_label,
    _principal_angle_stats, _rep_dir, _rotated_init_fit_path,
    _rowspace_principal_angle_stats, _save_fit,
)

def _dtype_from_name(name: str) -> torch.dtype:
    if name in {"float32", "torch.float32"}:
        return torch.float32
    if name in {"float64", "double", "torch.float64"}:
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name!r}")


def _fit_method(
    name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    params: dict[str, Any],
    *,
    save_dir: Path | None = None,
    progress_callback=None,
):
    if name == "zero_intercept_ols":
        return fit_zero_intercept_ols(x, y, ridge=float(params.get("ridge", 1e-8)))
    if name == "coordinatewise_huber":
        return fit_coordinatewise_huber(
            x,
            y,
            tau=float(params.get("tau", 1.345)),
            max_iter=int(params.get("max_iter", 100)),
            ridge=float(params.get("ridge", 1e-8)),
            intercept_mode=str(params.get("intercept_mode", "zero")),
        )
    if name == "radial_huber":
        tau_value = params.get("tau")
        return fit_radial_huber(
            x,
            y,
            tau=None if tau_value is None else float(tau_value),
            tau_multiplier=float(params.get("tau_multiplier", 1.345)),
            max_iter=int(params.get("max_iter", 100)),
            ridge=float(params.get("ridge", 1e-8)),
            intercept_mode=str(params.get("intercept_mode", "zero")),
        )
    if name == "coordinatewise_asm":
        from otrank.estimators import ASMConfig, fit_coordinatewise_asm_with_inference

        return fit_coordinatewise_asm_with_inference(
            x, y, ASMConfig(**dict(params))
        )
    if name == "mcqr_ot_lens":
        from otrank.estimators import MCQROTConfig, fit_mcqr_ot_lens

        return fit_mcqr_ot_lens(x, y, MCQROTConfig(**dict(params)))
    if name == "ot_rank_pnw_fullrisk":
        from otrank.estimators import FullRiskPNWLinearConfig, FullRiskPNWLinearEstimator
        from otrank.ot_maps import PNWEntropicOTConfig

        pnw_params = dict(params.get("pnw", {}))
        config = FullRiskPNWLinearConfig(
            pnw=PNWEntropicOTConfig(**pnw_params),
            steps=int(params.get("steps", params.get("max_iter", 500))),
            lr=float(params.get("lr", 1e-4)),
            optimizer=str(params.get("optimizer", "adam")),
            eval_steps=tuple(int(value) for value in params.get("eval_iters", params.get("eval_steps", [500]))),
            trace_every=int(params.get("trace_every", 50)),
            checkpoint_every=int(params.get("checkpoint_every", 50)),
            init_ridge=float(params.get("init_ridge", 1e-8)),
            intercept_mode=str(params.get("intercept_mode", "zero")),
            resume=bool(params.get("resume", True)),
            checkpoint_scores=bool(params.get("checkpoint_scores", False)),
            trace_first_step=bool(params.get("trace_first_step", False)),
            trace_tensor_stats=bool(params.get("trace_tensor_stats", False)),
        )
        return FullRiskPNWLinearEstimator(
            config=config,
            init_fit_path=params.get("init_fit_path"),
            save_dir=save_dir,
        ).fit(x, y)
    if name == "relu_pnw_fullrisk":
        from otrank.estimators import FullRiskPNWReLUConfig, FullRiskPNWReLUEstimator
        from otrank.ot_maps import PNWEntropicOTConfig

        pnw_params = dict(params.get("pnw", {}))
        config = FullRiskPNWReLUConfig(
            pnw=PNWEntropicOTConfig(**pnw_params),
            width=int(params.get("width", 16)),
            hidden_layers=int(params.get("hidden_layers", 3)),
            steps=int(params.get("steps", params.get("max_iter", 500))),
            lr=float(params.get("lr", 1e-4)),
            weight_decay=float(params.get("weight_decay", 1e-3)),
            optimizer=str(params.get("optimizer", "adam")),
            eval_steps=tuple(int(value) for value in params.get("eval_iters", params.get("eval_steps", [500]))),
            trace_every=int(params.get("trace_every", 50)),
            checkpoint_every=int(params.get("checkpoint_every", 50)),
            intercept_mode=str(params.get("intercept_mode", "zero")),
            seed=int(params.get("seed", 20260721)),
            resume=bool(params.get("resume", True)),
            init_mse_steps=int(params.get("init_mse_steps", 0)),
            init_mse_lr=float(params.get("init_mse_lr", 1e-3)),
            trace_first_step=bool(params.get("trace_first_step", False)),
            trace_tensor_stats=bool(params.get("trace_tensor_stats", False)),
        )
        return FullRiskPNWReLUEstimator(
            config=config,
            save_dir=save_dir,
            init_fit_path=params.get("init_fit_path"),
        ).fit(x, y)
    if name in {"relu_least_squares", "relu_coordinatewise_huber", "relu_radial_huber"}:
        from otrank.estimators import (
            ReLURegressionConfig,
            fit_relu_coordinatewise_huber,
            fit_relu_least_squares,
            fit_relu_radial_huber,
        )

        config = ReLURegressionConfig(
            width=int(params.get("width", 16)),
            hidden_layers=int(params.get("hidden_layers", 3)),
            steps=int(params.get("steps", 1000)),
            lr=float(params.get("lr", 1e-3)),
            weight_decay=float(params.get("weight_decay", 1e-3)),
            optimizer=str(params.get("optimizer", "adam")),
            momentum=float(params.get("momentum", 0.0)),
            output_init=str(params.get("output_init", "default")),
            intercept_mode=str(params.get("intercept_mode", "center")),
            seed=int(params.get("seed", 20260721)),
            huber_tau=float(params.get("huber_tau", 1.345)),
            radial_huber_tau=None if params.get("radial_huber_tau") is None else float(params["radial_huber_tau"]),
            radial_huber_tau_multiplier=float(params.get("radial_huber_tau_multiplier", 1.345)),
            trace_every=int(params.get("trace_every", 0)),
        )
        if name == "relu_least_squares":
            return fit_relu_least_squares(x, y, config)
        if name == "relu_coordinatewise_huber":
            return fit_relu_coordinatewise_huber(x, y, config)
        return fit_relu_radial_huber(x, y, config)
    if name == "relu_coordinatewise_cqr":
        from otrank.estimators import ReLUCQRConfig, fit_relu_coordinatewise_cqr

        levels = tuple(float(value) for value in params.get("quantile_levels", [level / 21.0 for level in range(1, 21)]))
        return fit_relu_coordinatewise_cqr(
            x,
            y,
            ReLUCQRConfig(
                width=int(params.get("width", 16)),
                hidden_layers=int(params.get("hidden_layers", 3)),
                steps=int(params.get("steps", 1000)),
                lr=float(params.get("lr", 1e-3)),
                weight_decay=float(params.get("weight_decay", 1e-3)),
                quantile_levels=levels,
                output_init=str(params.get("output_init", "default")),
                intercept_mode=str(params.get("intercept_mode", "center")),
                seed=int(params.get("seed", 20260721)),
            ),
        )
    if name == "relu_coordinatewise_asm":
        from otrank.estimators import ReLUASMConfig, ReLURegressionConfig, fit_relu_coordinatewise_asm

        pilot_params = dict(params.get("pilot", {}))
        width = int(params.get("width", 16))
        hidden_layers = int(params.get("hidden_layers", 3))
        pilot = ReLURegressionConfig(
            width=width,
            hidden_layers=hidden_layers,
            steps=int(pilot_params.get("steps", params.get("pilot_steps", 1000))),
            lr=float(pilot_params.get("lr", params.get("pilot_lr", 1e-3))),
            weight_decay=float(pilot_params.get("weight_decay", params.get("weight_decay", 1e-3))),
            optimizer=str(pilot_params.get("optimizer", params.get("optimizer", "adam"))),
            momentum=float(pilot_params.get("momentum", params.get("momentum", 0.0))),
            output_init=str(pilot_params.get("output_init", params.get("output_init", "default"))),
            intercept_mode=str(pilot_params.get("intercept_mode", params.get("intercept_mode", "center"))),
            seed=int(pilot_params.get("seed", params.get("seed", 20260721))),
            huber_tau=float(pilot_params.get("huber_tau", params.get("huber_tau", 1.345))),
        )
        config = ReLUASMConfig(
            pilot=pilot,
            width=width,
            hidden_layers=hidden_layers,
            steps=int(params.get("steps", 1000)),
            lr=float(params.get("lr", 5e-4)),
            weight_decay=float(params.get("weight_decay", 1e-3)),
            optimizer=str(params.get("optimizer", "adam")),
            momentum=float(params.get("momentum", 0.0)),
            alt_iter=int(params.get("alt_iter", 1)),
            score_grid_size=int(params.get("score_grid_size", 3000)),
            kernel_points=int(params.get("kernel_points", 32768)),
            bandwidth=params.get("bandwidth", "nrd0"),
            seed=int(params.get("seed", 20260721)),
            trace_every=int(params.get("trace_every", 0)),
        )
        return fit_relu_coordinatewise_asm(x, y, config)
    raise ValueError(f"Unknown method: {name}")


def _make_simulation_from_config(
    config: dict[str, Any],
    *,
    seed: int,
    noise: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
):
    data_cfg = config["data"]
    runtime_cfg = config.get("runtime", {})
    generation_device = torch.device("cpu") if bool(runtime_cfg.get("generate_on_cpu", True)) else device
    dgp = str(data_cfg.get("dgp", "linear")).lower()
    sim_kwargs = dict(
        n_train=int(data_cfg["n_train"]),
        n_test=int(data_cfg.get("n_test", 2000)),
        p=int(data_cfg["p"]),
        d=int(data_cfg["d"]),
        noise=noise,
        seed=int(seed),
        x_rho=float(data_cfg.get("x_rho", 0.0)),
        x_center=bool(data_cfg.get("x_center", True)),
        mu_nonzero=int(data_cfg.get("mu_nonzero", 0)),
        mu_value=float(data_cfg.get("mu_value", 0.0)),
        signal=float(data_cfg.get("signal", 1.0)),
        coef_style=str(data_cfg.get("coef_style", "gaussian")),
        singular_decay=float(data_cfg.get("singular_decay", 0.7)),
        noise_scale=float(data_cfg.get("noise_scale", 1.0)),
        device=generation_device,
        dtype=dtype,
    )
    if dgp == "linear":
        sim = make_linear_simulation(coef_rank=data_cfg.get("coef_rank"), **sim_kwargs)
    elif dgp == "gaussian_nonlinear":
        sim = make_nonlinear_simulation(
            n_train=int(data_cfg["n_train"]),
            n_test=int(data_cfg.get("n_test", 5000)),
            p=int(data_cfg["p"]),
            d=int(data_cfg["d"]),
            noise=noise,
            seed=int(seed),
            x_rho=float(data_cfg.get("x_rho", 0.0)),
            signal=float(data_cfg.get("signal", 1.0)),
            noise_scale=float(data_cfg.get("noise_scale", 1.0)),
            device=generation_device,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown DGP: {dgp!r}")
    if generation_device == device:
        return sim
    values = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in vars(sim).items()
    }
    return type(sim)(**values)


def _run_one(
    config: dict[str, Any],
    rep_index: int,
    seed: int,
    noise: dict[str, Any],
    method: dict[str, Any],
    *,
    job_index: int,
    out_dir: Path,
    progress_callback=None,
) -> list[dict[str, Any]]:
    data_cfg = config["data"]
    runtime_cfg = config.get("runtime", {})
    device = torch.device(runtime_cfg.get("device", "cpu"))
    dtype = _dtype_from_name(runtime_cfg.get("dtype", "float64"))
    torch.set_num_threads(int(runtime_cfg.get("torch_num_threads", 1)))
    dgp = str(data_cfg.get("dgp", "linear")).lower()
    sim = _make_simulation_from_config(config, seed=seed, noise=noise, device=device, dtype=dtype)
    method_name = method["name"]
    method_label = _method_label(method)
    noise_label = _noise_label(noise)
    method_params = dict(method.get("params", {}))
    if bool(config.get("relu_init_seed_by_rep", False)) and method_name.startswith("relu_"):
        init_seed = int(method_params.get("seed", 20260721)) + int(rep_index) - 1
        method_params["seed"] = init_seed
        if isinstance(method_params.get("pilot"), dict):
            method_params["pilot"] = {**method_params["pilot"], "seed": init_seed}
    fit_save_dir = _rep_dir(out_dir, data_cfg, noise, method, rep_index)
    completed_rows = _load_completed_job_rows(fit_save_dir, method_name, method_params)
    if completed_rows is not None:
        return completed_rows
    fit_save_dir.mkdir(parents=True, exist_ok=True)
    init_fit_path = _find_init_fit_path(out_dir, data_cfg, noise, rep_index, method_params)
    if (
        method_name.startswith("ot_rank") or method_name == "relu_pnw_fullrisk"
    ) and "init_fit_path" not in method_params and init_fit_path is not None:
        method_params["init_fit_path"] = str(init_fit_path)
    from otrank.data.rotations import rotation_spec
    rotation_enabled = bool(config.get("rotation_diagnostic", True)) or bool(method_params.get("ot_rank_rotation_diagnostic", False))
    recorded_rotation = rotation_spec(seed+10000, int(data_cfg["d"]), device, dtype) if rotation_enabled else None
    (fit_save_dir / "job_config.json").write_text(
        json.dumps(
            {
                "job_index": job_index,
                "rep_index": rep_index,
                "seed": seed,
                "noise": noise,
                "method": method,
                "effective_method_params": method_params,
                "data": data_cfg,
                "rotation_replay": recorded_rotation,
                "runtime": {"device": str(device), "dtype": str(dtype), "torch_version": torch.__version__},
            },
            indent=2,
            sort_keys=True,
        )
    )
    def make_row(
        row_fit,
        row_label: str,
        row_runtime: float,
        *,
        iteration: int | None = None,
        include_rotation: bool = True,
    ) -> dict[str, Any]:
        pred = row_fit.predict(sim.x_test)
        has_coef = hasattr(row_fit, "coef")
        row = {
        "seed": seed,
        "rep_index": rep_index,
        "method": row_label,
        "method_folder": method_label,
        "method_family": method_name,
        "noise": noise_label,
        "noise_family": noise["name"],
        "dgp": dgp,
        "n_train": int(data_cfg["n_train"]),
        "n_test": int(data_cfg.get("n_test", 2000)),
        "p": int(data_cfg["p"]),
        "d": int(data_cfg["d"]),
        "prediction_mse": prediction_mse(pred, sim.f_test),
        "prediction_mse_per_dim": prediction_mse_per_dim(pred, sim.f_test),
        "prediction_rmse_per_dim": prediction_rmse_per_dim(pred, sim.f_test),
        "prediction_mae": prediction_mae(pred, sim.f_test),
        "translation_adjusted_mse": translation_adjusted_mse(pred, sim.f_test),
        "centered_prediction_mse": centered_prediction_mse(pred, sim.f_test),
        "mean_bias": mean_bias(pred, sim.f_test),
        "coef_error": coefficient_error(row_fit.coef, sim.coef) if has_coef else float("nan"),
        "coef_angle": frobenius_angle(row_fit.coef, sim.coef) if has_coef else float("nan"),
        "runtime_sec": row_runtime,
        }
        if iteration is not None:
            row["iteration"] = int(iteration)

        estimated_jacobian = None
        if bool(config.get("jacobian_diagnostic", True)) and hasattr(row_fit, "jacobian"):
            estimated_jacobian = row_fit.jacobian(sim.x_test)
            if dgp == "gaussian_nonlinear":
                truth_jacobian = gaussian_nonlinear_jacobian(
                    sim.x_test,
                    signal=float(data_cfg.get("signal", 1.0)),
                )
                row["jacobian_estimation_rmse_per_entry"] = jacobian_rmse_per_entry(
                    estimated_jacobian,
                    truth_jacobian,
                )

        if include_rotation and bool(config.get("rotation_diagnostic", True)):
            generator = make_generator(seed + 10_000, device=device)
            q = random_orthogonal(sim.y_train.shape[1], generator, device=device, dtype=dtype)
            rotation_params = dict(method_params)
            rotation_params.pop("init_fit_path", None)
            fit_rot = _fit_method(
                method_name,
                sim.x_train,
                sim.y_train @ q.T,
                rotation_params,
                progress_callback=progress_callback,
            )
            pred_rot_back = fit_rot.predict(sim.x_test) @ q
            row["rotation_pred_gap"] = rotation_prediction_gap(pred, pred_rot_back)
            row["rotation_pred_mse_per_dim"] = rotation_prediction_mse_per_dim(pred, pred_rot_back)
            row["rotation_pred_rmse_per_dim"] = rotation_prediction_rmse_per_dim(pred, pred_rot_back)
            row["rotation_pred_angle"] = prediction_space_angle(pred, pred_rot_back)
            pred_angle_max, pred_fro_sin = _principal_angle_stats(pred, pred_rot_back)
            row["rotation_principal_angle_max"] = pred_angle_max
            row["rotation_principal_angle_fro_sin"] = pred_fro_sin
            if estimated_jacobian is not None and hasattr(fit_rot, "jacobian"):
                rotated_jacobian = fit_rot.jacobian(sim.x_test)
                rotated_jacobian_back = torch.einsum("ab,nbp->nap", q.T, rotated_jacobian)
                row["rotation_jacobian_rmse_per_entry"] = jacobian_rmse_per_entry(
                    rotated_jacobian_back,
                    estimated_jacobian,
                )
                row["rotation_jacobian_relative_gap"] = jacobian_relative_gap(
                    rotated_jacobian_back,
                    estimated_jacobian,
                )
                jacobian_angles = jacobian_rowspace_diagnostics(estimated_jacobian, rotated_jacobian_back)
                row["rotation_jacobian_rowspace_max_angle_mean"] = jacobian_angles["max_angle_mean"]
                row["rotation_jacobian_rowspace_max_angle_median"] = jacobian_angles["max_angle_median"]
                row["rotation_jacobian_rowspace_max_angle_q90"] = jacobian_angles["max_angle_q90"]
                row["rotation_jacobian_rowspace_fro_sin_mean"] = jacobian_angles["fro_sin_mean"]
                row["rotation_jacobian_projection_distance"] = jacobian_angles["projection_distance"]
                row["rotation_jacobian_full_rank_fraction"] = jacobian_angles["full_rank_fraction"]
                row["rotation_jacobian_rank_mismatch_fraction"] = jacobian_angles["rank_mismatch_fraction"]
            if has_coef and hasattr(fit_rot, "coef"):
                from otrank.metrics import rotation_coefficient_max_error
                row["e_max"] = rotation_coefficient_max_error(row_fit.coef, fit_rot.coef, q)
                coef_rot_back = q.T @ fit_rot.coef
                response_max, response_fro_sin = _principal_angle_stats(row_fit.coef, coef_rot_back)
                row["rotation_response_principal_angle_max"] = response_max
                row["rotation_response_principal_angle_fro_sin"] = response_fro_sin
                coef_max, coef_fro_sin = _rowspace_principal_angle_stats(row_fit.coef, coef_rot_back)
                row["rotation_coef_principal_angle_max"] = coef_max
                row["rotation_coef_principal_angle_fro_sin"] = coef_fro_sin
                row["rotation_coef_weighted_sine_discrepancy"] = weighted_sine_discrepancy(
                    row_fit.coef,
                    coef_rot_back,
                )
        return row

    start = time.perf_counter()

    fit = _fit_method(
        method_name,
        sim.x_train,
        sim.y_train,
        method_params,
        save_dir=fit_save_dir,
        progress_callback=progress_callback,
    )
    runtime = time.perf_counter() - start
    _save_fit(fit_save_dir / "fit.pt", fit)

    eval_iters = [int(value) for value in method_params.get("eval_iters", [])]
    rows: list[dict[str, Any]] = []
    if eval_iters and _is_iteration_method(method_name):
        if bool(method_params.get("ot_rank_rotation_diagnostic", False)):
            rotated_dir = fit_save_dir / "rotated"
            required_rotated = [
                rotated_dir / f"iteration_{iteration:03d}" / "state.pt"
                for iteration in eval_iters
            ]
            if any(not path.exists() for path in required_rotated):
                generator = make_generator(seed + 10_000, device=device)
                q = random_orthogonal(sim.y_train.shape[1], generator, device=device, dtype=dtype)
                rotation_params = dict(method_params)
                rotation_params.pop("init_fit_path", None)
                rotation_params["ot_rank_rotation_diagnostic"] = False
                if init_fit_path is not None:
                    rotated_init_path = _rotated_init_fit_path(
                        source_path=init_fit_path,
                        target_path=rotated_dir / f"rotated_init_from_{init_fit_path.parent.name}.pt",
                        q=q,
                        device=device,
                        dtype=dtype,
                    )
                    if rotated_init_path is not None:
                        rotation_params["init_fit_path"] = str(rotated_init_path)
                _fit_method(
                    method_name,
                    sim.x_train,
                    sim.y_train @ q.T,
                    rotation_params,
                    save_dir=rotated_dir,
                    progress_callback=progress_callback,
                )
        for iteration in eval_iters:
            iter_fit = _load_ot_rank_iteration_fit(fit_save_dir, iteration, device=device)
            row = make_row(
                iter_fit,
                f"{method_label}_iter{iteration}",
                runtime,
                iteration=iteration,
                include_rotation=False,
            )
            if bool(method_params.get("ot_rank_rotation_diagnostic", False)):
                rotated_dir = fit_save_dir / "rotated"
                if rotated_dir.exists():
                    generator = make_generator(seed + 10_000, device=device)
                    q = random_orthogonal(sim.y_train.shape[1], generator, device=device, dtype=dtype)
                    fit_rot = _load_ot_rank_iteration_fit(rotated_dir, iteration, device=device)
                    pred = iter_fit.predict(sim.x_test)
                    pred_rot_back = fit_rot.predict(sim.x_test) @ q
                    row["rotation_pred_gap"] = rotation_prediction_gap(pred, pred_rot_back)
                    row["rotation_pred_mse_per_dim"] = rotation_prediction_mse_per_dim(pred, pred_rot_back)
                    row["rotation_pred_rmse_per_dim"] = rotation_prediction_rmse_per_dim(pred, pred_rot_back)
                    row["rotation_pred_angle"] = prediction_space_angle(pred, pred_rot_back)
                    pred_angle_max, pred_fro_sin = _principal_angle_stats(pred, pred_rot_back)
                    row["rotation_principal_angle_max"] = pred_angle_max
                    row["rotation_principal_angle_fro_sin"] = pred_fro_sin
                    if hasattr(iter_fit, "coef") and hasattr(fit_rot, "coef"):
                        from otrank.metrics import rotation_coefficient_max_error
                        row["e_max"] = rotation_coefficient_max_error(iter_fit.coef, fit_rot.coef, q)
                        coef_rot_back = q.T @ fit_rot.coef
                        response_max, response_fro_sin = _principal_angle_stats(iter_fit.coef, coef_rot_back)
                        row["rotation_response_principal_angle_max"] = response_max
                        row["rotation_response_principal_angle_fro_sin"] = response_fro_sin
                        coef_max, coef_fro_sin = _rowspace_principal_angle_stats(iter_fit.coef, coef_rot_back)
                        row["rotation_coef_principal_angle_max"] = coef_max
                        row["rotation_coef_principal_angle_fro_sin"] = coef_fro_sin
                        row["rotation_coef_weighted_sine_discrepancy"] = weighted_sine_discrepancy(
                            iter_fit.coef,
                            coef_rot_back,
                        )
            rows.append(row)
            (fit_save_dir / f"metrics_iter{iteration:03d}.json").write_text(json.dumps(row, indent=2, sort_keys=True))
    else:
        rows.append(make_row(fit, method_label, runtime, include_rotation=not _is_iteration_method(method_name)))
        (fit_save_dir / "metrics.json").write_text(json.dumps(rows[0], indent=2, sort_keys=True))
    return rows


def _fit_from_state(state_path: Path, device: torch.device):
    from otrank.estimators.baselines import LinearModelFit

    state = _load_checkpoint(state_path, device)
    if str(state.get("model_type", "")) == "relu_feedforward":
        from otrank.estimators.neural_pnw import relu_fit_from_state

        dtype = state.get("x_mean", torch.empty((), dtype=torch.float32)).dtype
        return relu_fit_from_state(state, device=device, dtype=dtype)
    return LinearModelFit(
        coef=state["coef"].to(device=device),
        x_mean=state["x_mean"].to(device=device),
        y_mean=state["y_mean"].to(device=device),
        method="ot_rank_iteration",
    )


def _load_ot_rank_iteration_fit(run_dir: Path, iteration: int, *, device: torch.device):
    return _fit_from_state(run_dir / f"iteration_{iteration:03d}" / "state.pt", device)
