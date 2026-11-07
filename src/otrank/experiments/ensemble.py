"""Reference-member construction and ensemble aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from otrank.data.rotations import replay_rotation
from otrank.metrics import (
    centered_prediction_mse, coefficient_error, frobenius_angle, mean_bias,
    prediction_mae, prediction_mse, prediction_mse_per_dim,
    prediction_rmse_per_dim, prediction_space_angle, rotation_prediction_gap,
    rotation_prediction_mse_per_dim, rotation_prediction_rmse_per_dim,
    translation_adjusted_mse, weighted_sine_discrepancy,
    rotation_coefficient_max_error,
)

from .core import _dtype_from_name, _make_simulation_from_config
from .layout import (
    _atomic_torch_save, _atomic_write_json, _canonical_method_label,
    _ensemble_metrics_path, _ensemble_state_path, _load_checkpoint,
    _load_completed_job_rows, _method_label, _noise_label,
    _principal_angle_stats, _rep_dir, _rowspace_principal_angle_stats,
)

def _resource_seed_label(base_label: str, reference_seed: int) -> str:
    return f"{base_label}_refseed{int(reference_seed)}"

def _resource_ensemble_label(base_label: str) -> str:
    return f"{base_label}_refensemble"

def _resource_clone_ot_method(
    method: dict[str, Any],
    *,
    reference_seed: int,
    reference_index: int,
    cpu_checkpoint_every: int | None = None,
) -> dict[str, Any]:
    cloned = json.loads(json.dumps(method))
    base_label = _method_label(cloned)
    cloned["label"] = _resource_seed_label(base_label, reference_seed)
    cloned["_canonical_base_label"] = base_label
    cloned["_reference_index"] = int(reference_index)
    cloned["_reference_seed"] = int(reference_seed)
    params = dict(cloned.get("params", {}))
    pnw = dict(params.get("pnw", {}))
    pnw["seed"] = int(reference_seed)
    params["pnw"] = pnw
    if cpu_checkpoint_every is not None:
        params["checkpoint_every"] = int(cpu_checkpoint_every)
    cloned["params"] = params
    return cloned

def _resource_eval_iters(method: dict[str, Any]) -> list[int]:
    params = dict(method.get("params", {}))
    values = params.get("eval_iters", params.get("eval_steps", [params.get("steps", 500)]))
    return [int(value) for value in values]

def _resource_completed_rows(
    out_dir: Path,
    data_cfg: dict[str, Any],
    noise: dict[str, Any],
    rep_index: int,
    method: dict[str, Any],
) -> list[dict[str, Any]] | None:
    fit_save_dir = _rep_dir(out_dir, data_cfg, noise, method, int(rep_index))
    return _load_completed_job_rows(fit_save_dir, str(method.get("name", "")), dict(method.get("params", {})))

def _resource_load_linear_fit(path: Path, *, device: torch.device):
    from otrank.estimators.baselines import LinearModelFit

    state = _load_checkpoint(path, device)
    return LinearModelFit(
        coef=state["coef"].to(device=device),
        x_mean=state["x_mean"].to(device=device),
        y_mean=state["y_mean"].to(device=device),
        method=str(state.get("method", "resource_scheduler_fit")),
    )

def _resource_mean_fit(fits: list[Any], *, method: str):
    from otrank.estimators.baselines import LinearModelFit

    if not fits:
        raise ValueError("Cannot average an empty fit list")
    return LinearModelFit(
        coef=torch.stack([fit.coef for fit in fits], dim=0).mean(dim=0),
        x_mean=torch.stack([fit.x_mean for fit in fits], dim=0).mean(dim=0),
        y_mean=torch.stack([fit.y_mean for fit in fits], dim=0).mean(dim=0),
        method=method,
    )

def _resource_ensemble_row(
    *,
    config: dict[str, Any],
    out_dir: Path,
    rep_index: int,
    seed: int,
    noise: dict[str, Any],
    base_method: dict[str, Any],
    reference_seeds: list[int],
    k: int,
    iteration: int,
    runtime_sec: float,
    dtype_name: str,
) -> dict[str, Any]:
    device = torch.device("cpu")
    dtype = _dtype_from_name(dtype_name)
    sim = _make_simulation_from_config(config, seed=seed, noise=noise, device=device, dtype=dtype)
    q = None
    recorded_rotation = None
    base_label = _method_label(base_method)
    fits = []
    native_rotated_fits = []
    rotated_back_fits = []
    for reference_index, reference_seed in enumerate(reference_seeds[: int(k)], 1):
        method = _resource_clone_ot_method(
            base_method,
            reference_seed=int(reference_seed),
            reference_index=int(reference_index),
        )
        run_dir = _rep_dir(out_dir, config["data"], noise, method, int(rep_index))
        job = json.loads((run_dir/"job_config.json").read_text())
        spec = job.get("rotation_replay", config.get("rotation_replay"))
        if not isinstance(spec, dict):
            raise ValueError("Member rotation seed/runtime metadata are missing; do not rebuild a CUDA rotation on CPU.")
        if spec.get("seed") != int(seed)+10000 or spec.get("dimension") != sim.y_train.shape[1]:
            raise ValueError("member rotation metadata do not match this repetition")
        if recorded_rotation is None:
            recorded_rotation = spec
            q = replay_rotation(spec, output_device=device)
        elif spec != recorded_rotation:
            raise ValueError("ensemble members have inconsistent rotation replay metadata")
        fit = _resource_load_linear_fit(run_dir / f"iteration_{int(iteration):03d}" / "state.pt", device=device)
        fit_rot = _resource_load_linear_fit(run_dir / "rotated" / f"iteration_{int(iteration):03d}" / "state.pt", device=device)
        fits.append(fit)
        native_rotated_fits.append(fit_rot)
        from otrank.estimators.baselines import LinearModelFit

        rotated_back_fits.append(
            LinearModelFit(
                coef=q.T @ fit_rot.coef,
                x_mean=fit_rot.x_mean,
                y_mean=fit_rot.y_mean @ q,
                method="rotated_back",
            )
        )
    ensemble_label = _resource_ensemble_label(base_label)
    fit = _resource_mean_fit(fits, method=f"{ensemble_label}_K{k}")
    fit_rot_back = _resource_mean_fit(rotated_back_fits, method=f"{ensemble_label}_K{k}_rotated_back")
    pred = fit.predict(sim.x_test)
    pred_rot_back = fit_rot_back.predict(sim.x_test)
    coef_max, coef_fro_sin = _rowspace_principal_angle_stats(fit.coef, fit_rot_back.coef)
    pred_max, pred_fro_sin = _principal_angle_stats(pred, pred_rot_back)
    row = {
        "seed": int(seed),
        "rep_index": int(rep_index),
        "method": f"{ensemble_label}_K{int(k)}_iter{int(iteration)}",
        "method_folder": base_label,
        "method_family": "ot_rank_pnw_reference_ensemble",
        "noise": _noise_label(noise),
        "noise_family": str(noise.get("name", noise.get("label", "noise"))),
        "dgp": str(config["data"].get("dgp", "linear")),
        "n_train": int(config["data"]["n_train"]),
        "n_test": int(config["data"].get("n_test", 2000)),
        "p": int(config["data"]["p"]),
        "d": int(config["data"]["d"]),
        "iteration": int(iteration),
        "ensemble_k": int(k),
        "reference_seeds": ",".join(str(seed_value) for seed_value in reference_seeds[: int(k)]),
        "prediction_mse": prediction_mse(pred, sim.f_test),
        "prediction_mse_per_dim": prediction_mse_per_dim(pred, sim.f_test),
        "prediction_rmse_per_dim": prediction_rmse_per_dim(pred, sim.f_test),
        "prediction_mae": prediction_mae(pred, sim.f_test),
        "translation_adjusted_mse": translation_adjusted_mse(pred, sim.f_test),
        "centered_prediction_mse": centered_prediction_mse(pred, sim.f_test),
        "mean_bias": mean_bias(pred, sim.f_test),
        "coef_error": coefficient_error(fit.coef, sim.coef),
        "e_max": rotation_coefficient_max_error(fit.coef, _resource_mean_fit(native_rotated_fits, method="native_rotated_mean").coef, q),
        "coef_angle": frobenius_angle(fit.coef, sim.coef),
        "rotation_pred_gap": rotation_prediction_gap(pred, pred_rot_back),
        "rotation_pred_mse_per_dim": rotation_prediction_mse_per_dim(pred, pred_rot_back),
        "rotation_pred_rmse_per_dim": rotation_prediction_rmse_per_dim(pred, pred_rot_back),
        "rotation_pred_angle": prediction_space_angle(pred, pred_rot_back),
        "rotation_principal_angle_max": pred_max,
        "rotation_principal_angle_fro_sin": pred_fro_sin,
        "rotation_coef_principal_angle_max": coef_max,
        "rotation_coef_principal_angle_fro_sin": coef_fro_sin,
        "rotation_coef_weighted_sine_discrepancy": weighted_sine_discrepancy(
            fit.coef,
            fit_rot_back.coef,
        ),
        "runtime_sec": float(runtime_sec),
    }
    metrics_path = _ensemble_metrics_path(
        out_dir,
        config["data"],
        noise,
        base_method,
        int(rep_index),
    )
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text())
    else:
        payload = {
            "schema_version": 3,
            "method": _canonical_method_label(base_method),
            "rep_index": int(rep_index),
            "metrics": {},
        }
    payload.setdefault("metrics", {}).setdefault(f"K{int(k):03d}", {})[
        f"iter{int(iteration):03d}"
    ] = row
    _atomic_write_json(metrics_path, payload)
    _atomic_torch_save(
        _ensemble_state_path(
            out_dir,
            config["data"],
            noise,
            base_method,
            int(rep_index),
        ),
        {
            "schema_version": 3,
            "method": _canonical_method_label(base_method),
            "rep_index": int(rep_index),
            "completed_k": int(k),
            "iteration": int(iteration),
            "member_folders": [f"K{index:03d}" for index in range(1, int(k) + 1)],
            "original": {
                "coef": fit.coef.detach().cpu(),
                "x_mean": fit.x_mean.detach().cpu(),
                "y_mean": fit.y_mean.detach().cpu(),
                "method": fit.method,
            },
            "rotated_back": {
                "coef": fit_rot_back.coef.detach().cpu(),
                "x_mean": fit_rot_back.x_mean.detach().cpu(),
                "y_mean": fit_rot_back.y_mean.detach().cpu(),
                "method": fit_rot_back.method,
            },
        },
    )
    return row
