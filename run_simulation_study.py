"""Run the paper's final linear or ReLU simulation configuration.

This user-facing entry point reads a frozen paper config, then materializes the
lower-level jobs consumed by the simulation runner in ``src``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "simulation_linear.json"
PAPER_DIMENSIONS = {(20, 10)}
PAPER_SAMPLE_SIZES = {500, 1000}
PAPER_METHOD_KEYS = {
    "ls_free_intercept",
    "coordinatewise_huber_free_intercept",
    "radial_huber_free_intercept",
    "zero_intercept_ols",
    "coordinatewise_huber_tau1p345",
    "radial_huber_tau1p345_sqrtd",
    "coordinatewise_asm_zero_intercept",
    "cw_cqr_20levels",
    "mcqr_ot_lens",
    "mcqr_gaussian",
    "mcqr_hypercube",
    "ot_rank_ols_init",
    "ot_rank_huber_init",
    "ot_rank_radial_huber_init",
    "ot_rank_asm_init",
    "ot_rank_mcqr_init",
    "relu_pnw_fullrisk",
    "relu_least_squares",
    "relu_coordinatewise_huber",
    "relu_radial_huber",
    "relu_coordinatewise_asm",
    "relu_coordinatewise_cqr",
}


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value)).strip("_") or "simulation"


def build_cw_cqr_protocol(config: dict, *, n_train: int, p: int, d: int) -> dict:
    """Build the public CW-CQR handoff from the authoritative paper config."""

    dgp = dict(config.get("dgp", {}))
    ensemble = dict(config.get("ot_rank_ensemble", {}))
    return {
        "coefficient_generation": {
            "name": str(dgp.get("name", "linear")),
            "coef_style": str(dgp.get("coef_style", "gaussian")),
            "mu_nonzero": int(dgp.get("mu_nonzero", 0)),
            "mu_value": float(dgp.get("mu_value", 0.0)),
            "singular_decay": float(dgp.get("singular_decay", 0.7)),
            "x_center": bool(dgp.get("x_center", True)),
        },
        "formal_config": deepcopy(config),
        "intercept": "CW-CQR uses quantile-specific nuisance intercepts; only common slopes are compared",
        "method_labels": list(config.get("paper_methods", config.get("methods", []))),
        "rotation_convention": (
            "Response rotations use the deterministic seed+10000 orthogonal matrix "
            "and are mapped back before row-space metrics."
        ),
        "seed_convention": {
            "seed_start": int(config.get("seed_start", 20261001)),
            "rep_r_seed": f"{int(config.get('seed_start', 20261001))} + (rep_index-1)",
            "reference_seed_start": int(
                ensemble.get("reference_seed_start", 20260721)
            ),
        },
        "target": {
            "R": int(config["reps"]),
            "n_train": int(n_train),
            "p": int(p),
            "d": int(d),
            "noises": [str(item["label"]) for item in config["noises"]],
            "cw_cqr_quantiles": "tau_l=l/21, l=1,...,20",
        },
        "x_center": bool(dgp.get("x_center", True)),
    }


NOISES = [
    {
        "label": "A",
        "name": "low_rank_direction_outlier",
        "params": {"eps": 0.1, "rank": 1, "scale": 30.0, "sigma": 1.0, "standardize": "global", "tail": "centered_lognormal"},
    },
    {
        "label": "B",
        "name": "low_rank_direction_outlier",
        "params": {"alpha": 2.5, "eps": 0.1, "rank": 3, "scale": 20.0, "standardize": "global", "tail": "centered_pareto"},
    },
    {
        "label": "C",
        "name": "eight_cluster",
        "params": {"scale": 1.0, "sigma": 0.3, "standardize": "global"},
    },
]


def _ot_method(label: str, init_label: str, ot: dict) -> dict:
    label = str(ot.get("label", label))
    estimator = str(ot.get("ot_estimator", "pnw")).lower()
    backend = str(ot.get("backend", "dense_triton")).lower()
    if estimator != "pnw":
        raise ValueError("main simulation currently supports only ot_estimator='pnw'")
    if backend not in {"dense_pytorch", "dense_triton", "torch", "implicit_triton_loss_cached_ab_gemm_add"}:
        raise ValueError("main simulation currently supports only dense_pytorch or dense_triton PNW backends")
    return {
        "label": label,
        "name": "ot_rank_pnw_fullrisk",
        "requires_fit": [init_label],
        "params": {
            "checkpoint_every": int(ot.get("checkpoint_every", 50)),
            "eval_iters": [int(value) for value in ot.get("eval_iters", [ot.get("steps", 100)])],
            "init_fit_label": init_label,
            "init_cache_dirs": list(ot.get("init_cache_dirs", [])),
            "intercept_mode": "zero",
            "require_existing_init": bool(ot.get("require_existing_init", False)),
            "lr": float(ot.get("lr", 1e-4)),
            "optimizer": "adam",
            "pnw": {
                "ot_estimator": str(ot.get("ot_estimator", "pnw")),
                "epsilon": float(ot.get("epsilon", 0.10)),
                "m_ref": int(ot.get("m_ref", 131072)),
                "reference_distribution": str(ot.get("reference_distribution", "uniform_ball")),
                "reference_radius": float(ot.get("reference_radius", 1.0)),
                "seed": int(ot.get("reference_seed", 20260721)),
                "sinkhorn_iters": int(ot.get("sinkhorn_iters", 1)),
                "chunk_size": ot.get("chunk_size"),
                "topk_barycentric": ot.get("topk_barycentric"),
                "backend": str(ot.get("backend", "dense_triton")),
                "implicit_solve_dtype": str(ot.get("implicit_solve_dtype", "float32")),
                "implicit_gmres_iters": int(ot.get("implicit_gmres_iters", 30)),
                "implicit_gmres_tol": float(ot.get("implicit_gmres_tol", 1e-6)),
                "implicit_ridge": float(ot.get("implicit_ridge", 1e-8)),
                "implicit_chunk_size": int(ot.get("implicit_chunk_size", 1024)),
                "implicit_cache_dtype": str(ot.get("implicit_cache_dtype", "float32")),
                "sinkhorn_warm_start": bool(ot.get("sinkhorn_warm_start", True)),
                "triton_direct_block_n": int(ot.get("triton_direct_block_n", 8)),
                "triton_direct_block_m": int(ot.get("triton_direct_block_m", 1024)),
                "triton_matvec_block_n": int(ot.get("triton_matvec_block_n", 32)),
                "triton_matvec_block_m": int(ot.get("triton_matvec_block_m", 256)),
                "triton_implicit_add_block_m": int(ot.get("triton_implicit_add_block_m", 256)),
                "triton_num_warps": int(ot.get("triton_num_warps", 8)),
            },
            "ot_rank_rotation_diagnostic": True,
            "resume": True,
            "steps": int(ot.get("steps", 100)),
            "trace_every": int(ot.get("trace_every", 50)),
            "trace_first_step": bool(ot.get("trace_first_step", False)),
            "trace_tensor_stats": bool(ot.get("trace_tensor_stats", False)),
        },
    }


def _relu_pnw_method(config: dict) -> dict:
    ot = deepcopy(dict(config.get("ot_rank", {})))
    _deep_update(ot, dict(config.get("relu_pnw", {})))
    estimator = str(ot.get("ot_estimator", "pnw")).lower()
    backend = str(ot.get("backend", "dense_triton")).lower()
    if estimator != "pnw":
        raise ValueError("ReLU PNW simulation currently supports only ot_estimator='pnw'")
    if backend not in {"dense_pytorch", "dense_triton", "torch", "implicit_triton_loss_cached_ab_gemm_add"}:
        raise ValueError("ReLU PNW simulation currently supports only dense_pytorch or dense_triton PNW backends")
    init_label = str(ot.get("init_fit_label", "relu_coordinatewise_asm_huber_init"))
    return {
        "label": str(ot.get("label", "relu_pnw_fullrisk")),
        "name": "relu_pnw_fullrisk",
        "requires_fit": [init_label],
        "params": {
            "width": int(ot.get("width", 16)),
            "hidden_layers": int(ot.get("hidden_layers", 3)),
            "checkpoint_every": int(ot.get("checkpoint_every", 50)),
            "eval_iters": [int(value) for value in ot.get("eval_iters", [ot.get("steps", 100)])],
            "intercept_mode": str(ot.get("intercept_mode", "zero")),
            "lr": float(ot.get("lr", 1e-4)),
            "weight_decay": float(ot.get("weight_decay", 1e-3)),
            "optimizer": "adam",
            "seed": int(ot.get("seed", 20260721)),
            "init_fit_label": init_label,
            "init_cache_dirs": list(ot.get("init_cache_dirs", [])),
            "require_existing_init": bool(ot.get("require_existing_init", False)),
            "init_mse_steps": int(ot.get("init_mse_steps", 0)),
            "init_mse_lr": float(ot.get("init_mse_lr", 1e-3)),
            "pnw": {
                "ot_estimator": str(ot.get("ot_estimator", "pnw")),
                "epsilon": float(ot.get("epsilon", 0.10)),
                "m_ref": int(ot.get("m_ref", 131072)),
                "reference_distribution": str(ot.get("reference_distribution", "uniform_ball")),
                "reference_radius": float(ot.get("reference_radius", 1.0)),
                "seed": int(ot.get("reference_seed", ot.get("seed", 20260721))),
                "sinkhorn_iters": int(ot.get("sinkhorn_iters", 1)),
                "chunk_size": ot.get("chunk_size"),
                "topk_barycentric": ot.get("topk_barycentric"),
                "backend": str(ot.get("backend", "dense_triton")),
                "implicit_solve_dtype": str(ot.get("implicit_solve_dtype", "float32")),
                "implicit_gmres_iters": int(ot.get("implicit_gmres_iters", 30)),
                "implicit_gmres_tol": float(ot.get("implicit_gmres_tol", 1e-6)),
                "implicit_ridge": float(ot.get("implicit_ridge", 1e-8)),
                "implicit_chunk_size": int(ot.get("implicit_chunk_size", 1024)),
                "implicit_cache_dtype": str(ot.get("implicit_cache_dtype", "float32")),
                "sinkhorn_warm_start": bool(ot.get("sinkhorn_warm_start", True)),
                "triton_direct_block_n": int(ot.get("triton_direct_block_n", 8)),
                "triton_direct_block_m": int(ot.get("triton_direct_block_m", 1024)),
                "triton_matvec_block_n": int(ot.get("triton_matvec_block_n", 32)),
                "triton_matvec_block_m": int(ot.get("triton_matvec_block_m", 256)),
                "triton_implicit_add_block_m": int(ot.get("triton_implicit_add_block_m", 256)),
                "triton_num_warps": int(ot.get("triton_num_warps", 8)),
            },
            "ot_rank_rotation_diagnostic": bool(ot.get("ot_rank_rotation_diagnostic", True)),
            "resume": True,
            "steps": int(ot.get("steps", 100)),
            "trace_every": int(ot.get("trace_every", 50)),
            "trace_first_step": bool(ot.get("trace_first_step", False)),
            "trace_tensor_stats": bool(ot.get("trace_tensor_stats", False)),
        },
    }


def _deep_update(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = deepcopy(value)
    return dst


def _stage_configs(
    config: dict,
    *,
    requested_stage: str | None,
    requested_methods: list[str] | None,
) -> list[tuple[str | None, dict]]:
    """Expand a single paper config into its runtime-specific stages."""

    stages = dict(config.get("stages", {}))
    if not stages:
        if requested_stage is not None:
            raise ValueError("--stage is only valid for a staged configuration")
        materialized = deepcopy(config)
        if requested_methods:
            materialized["methods"] = list(requested_methods)
        return [(None, materialized)]

    if requested_stage is not None and requested_stage not in stages:
        raise ValueError(
            f"Unknown stage '{requested_stage}'. Available stages: {list(stages)}"
        )
    known_methods = set(config.get("methods", []))
    if requested_methods:
        unknown = sorted(set(requested_methods) - known_methods)
        if unknown:
            raise ValueError(f"Requested methods are absent from the paper config: {unknown}")

    common = deepcopy(config)
    common.pop("stages", None)
    output: list[tuple[str | None, dict]] = []
    for stage_name, stage_override in stages.items():
        if requested_stage is not None and stage_name != requested_stage:
            continue
        materialized = deepcopy(common)
        _deep_update(materialized, dict(stage_override))
        if requested_methods:
            stage_methods = set(materialized.get("methods", []))
            materialized["methods"] = [
                method for method in requested_methods if method in stage_methods
            ]
            if not materialized["methods"]:
                continue
        materialized["experiment"] = f"{config.get('experiment', 'simulation')}_{stage_name}"
        output.append((stage_name, materialized))
    if not output:
        raise ValueError("No stage contains the requested methods")
    return output


def _apply_runtime_overrides(config: dict, *, n_train: int, hardware_profile: str | None) -> dict:
    config = deepcopy(config)
    sample_overrides = dict(config.get("sample_size_overrides", {}))
    for key in (str(int(n_train)), int(n_train)):
        if key in sample_overrides:
            _deep_update(config, dict(sample_overrides[key]))
            break
    if hardware_profile:
        profiles = dict(config.get("hardware_profiles", {}))
        if hardware_profile not in profiles:
            raise ValueError(f"Unknown hardware profile '{hardware_profile}'. Available profiles: {sorted(profiles)}")
        _deep_update(config, dict(profiles[hardware_profile]))
    return config


def _methods(config: dict) -> list[dict]:
    requested = set(config.get("methods", []))
    asm_response_workers = int(config.get("asm_response_workers", 1))
    asm_numeric_threads = int(config.get("asm_numeric_threads", 1))
    base = {
        **{
            label: {
                "label": label,
                "name": "linear_free_intercept",
                "params": {"loss": loss, "ridge": 1e-8, "max_iter": 80, "tol": 1e-7,
                           "tau_multiplier": 1.345},
            }
            for label, loss in (
                ("ls_free_intercept", "ols"),
                ("coordinatewise_huber_free_intercept", "cw_huber"),
                ("radial_huber_free_intercept", "radial_huber"),
            )
        },
        "zero_intercept_ols": {"label": "zero_intercept_ols", "name": "zero_intercept_ols", "params": {"ridge": 1e-8}},
        "coordinatewise_huber_tau1p345": {
            "label": "coordinatewise_huber_tau1p345",
            "name": "coordinatewise_huber",
            "params": {"intercept_mode": "zero", "max_iter": 80, "ridge": 1e-8, "tau": 1.345},
        },
        "radial_huber_tau1p345_sqrtd": {
            "label": "radial_huber_tau1p345_sqrtd",
            "name": "radial_huber",
            "params": {"intercept_mode": "zero", "max_iter": 80, "ridge": 1e-8, "tau_multiplier": 1.345},
        },
        "coordinatewise_asm_zero_intercept": {
            "label": "coordinatewise_asm_zero_intercept",
            "name": "coordinatewise_asm",
            "params": {
                "alt_iter": 1,
                "betapilot": "OLS",
                "center_xy": False,
                "force_zero_intercept": True,
                "intercept_selection": "mean",
                "max_iter": 65,
                "numeric_threads": asm_numeric_threads,
                "response_workers": asm_response_workers,
                "symmetric": False,
            },
        },
        "mcqr_ot_lens": {
            "label": "mcqr_ot_lens",
            "name": "mcqr_ot_lens",
            "params": {"intercept_mode": "zero", "lr": 0.02, "max_iter": 200, "reference": "normal", "ridge": 1e-5, "seed": 2026, "show_progress": False},
        },
        "mcqr_gaussian": {
            "label": "mcqr_gaussian",
            "name": "mcqr_ot_lens",
            "params": {"intercept_mode": "zero", "lr": 0.02, "max_iter": 200, "optimizer": "adam", "reference": "normal", "ridge": 1e-5, "seed": 2026, "show_progress": False},
        },
        "mcqr_hypercube": {
            "label": "mcqr_hypercube",
            "name": "mcqr_ot_lens",
            "params": {"intercept_mode": "zero", "lr": 0.02, "max_iter": 200, "optimizer": "adam", "reference": "hypercube", "ridge": 1e-5, "seed": 2026, "show_progress": False},
        },
    }
    relu = dict(config.get("relu_baselines", {}))
    relu_common = {
        "width": int(relu.get("width", 16)),
        "hidden_layers": int(relu.get("hidden_layers", 3)),
        "steps": int(relu.get("steps", 1000)),
        "lr": float(relu.get("lr", 1e-3)),
        "weight_decay": float(relu.get("huber_weight_decay", 1e-3)),
        "optimizer": str(relu.get("optimizer", "adam")),
        "momentum": float(relu.get("momentum", 0.0)),
        "output_init": str(relu.get("output_init", "default")),
        "intercept_mode": str(relu.get("intercept_mode", "center")),
        "seed": int(relu.get("seed", 20260721)),
        "huber_tau": float(relu.get("huber_tau", 1.345)),
    }
    asm_relu = dict(config.get("relu_asm", {}))
    base.update(
        {
            "relu_least_squares": {
                "label": "relu_least_squares",
                "name": "relu_least_squares",
                "params": {
                    **deepcopy(relu_common),
                    "lr": float(relu.get("ls_lr", relu_common["lr"])),
                    "weight_decay": float(relu.get("ls_weight_decay", 1e-2)),
                },
            },
            "relu_coordinatewise_huber": {
                "label": "relu_coordinatewise_huber",
                "name": "relu_coordinatewise_huber",
                "params": deepcopy(relu_common),
            },
            "relu_radial_huber": {
                "label": "relu_radial_huber",
                "name": "relu_radial_huber",
                "params": {
                    **deepcopy(relu_common),
                    "lr": float(relu.get("radial_huber_lr", relu_common["lr"])),
                    "weight_decay": float(relu.get("radial_huber_weight_decay", 1e-3)),
                    "radial_huber_tau_multiplier": float(relu.get("radial_huber_tau_multiplier", 1.345)),
                },
            },
            "relu_coordinatewise_asm": {
                "label": "relu_coordinatewise_asm_huber_init",
                "name": "relu_coordinatewise_asm",
                "params": {
                    **deepcopy(relu_common),
                    "steps": int(asm_relu.get("steps", 1000)),
                    "lr": float(asm_relu.get("lr", 5e-4)),
                    "optimizer": str(asm_relu.get("optimizer", relu_common["optimizer"])),
                    "momentum": float(asm_relu.get("momentum", relu_common["momentum"])),
                    "alt_iter": int(asm_relu.get("alt_iter", 1)),
                    "score_grid_size": int(asm_relu.get("score_grid_size", 3000)),
                    "kernel_points": int(asm_relu.get("kernel_points", 32768)),
                    "bandwidth": asm_relu.get("bandwidth", "nrd0"),
                    "weight_decay": float(asm_relu.get("weight_decay", 1e-3)),
                    "pilot": deepcopy(relu_common),
                },
            },
            "relu_coordinatewise_cqr": {
                "label": "relu_coordinatewise_cqr",
                "name": "relu_coordinatewise_cqr",
                "params": {
                    **deepcopy(relu_common),
                    "lr": float(relu.get("cqr_lr", relu_common["lr"])),
                    "weight_decay": float(relu.get("cqr_weight_decay", 1e-3)),
                    "quantile_levels": [float(value) for value in relu.get("cqr_quantile_levels", [level / 21.0 for level in range(1, 21)])],
                },
            },
        }
    )
    ot = config.get("ot_rank", {})
    base.update(
        {
            "ot_rank_ols_init": _ot_method("ot_rank_ls_init", "zero_intercept_ols", ot),
            "ot_rank_huber_init": _ot_method("ot_rank_cw_huber_init", "coordinatewise_huber_tau1p345", ot),
            "ot_rank_radial_huber_init": _ot_method(
                "ot_rank_radial_huber_init",
                "radial_huber_tau1p345_sqrtd",
                ot,
            ),
            "ot_rank_asm_init": _ot_method("ot_rank_cw_asm_init", "coordinatewise_asm_zero_intercept", ot),
            "ot_rank_mcqr_init": _ot_method("ot_rank_mcqr_init", "mcqr_ot_lens", ot),
            "relu_pnw_fullrisk": _relu_pnw_method(config),
        }
    )
    ordered = config.get("methods", list(base))
    unknown = [name for name in ordered if name not in base]
    if unknown:
        raise ValueError(f"Unknown method key(s): {unknown}")
    return [deepcopy(base[name]) for name in ordered if not requested or name in requested]


def build_low_level_config(
    config: dict,
    *,
    n_train: int,
    p: int,
    d: int,
    workers: int,
    hardware_profile: str | None = None,
    noise_labels: list[str] | None = None,
    rep_start: int | None = None,
    rep_end: int | None = None,
    init_cache_dirs: list[str] | None = None,
    require_existing_init: bool | None = None,
) -> dict:
    config = _apply_runtime_overrides(config, n_train=int(n_train), hardware_profile=hardware_profile)
    all_seeds = [int(config.get("seed_start", 20260501)) + i for i in range(int(config["reps"]))]
    all_rep_indices = list(range(1, len(all_seeds) + 1))
    if rep_start is not None or rep_end is not None:
        lo = int(rep_start if rep_start is not None else 1)
        hi = int(rep_end if rep_end is not None else len(all_seeds))
        if lo < 1 or hi < lo or hi > len(all_seeds):
            raise ValueError(f"Invalid rep range [{lo}, {hi}] for {len(all_seeds)} repetitions")
        selected = [(rep, seed) for rep, seed in zip(all_rep_indices, all_seeds, strict=True) if lo <= rep <= hi]
        rep_indices = [rep for rep, _ in selected]
        seeds = [seed for _, seed in selected]
    else:
        rep_indices = all_rep_indices
        seeds = all_seeds
    if require_existing_init is not None:
        config.setdefault("ot_rank", {})["require_existing_init"] = bool(require_existing_init)
    if init_cache_dirs:
        config.setdefault("ot_rank", {})["init_cache_dirs"] = [str(path) for path in init_cache_dirs]
    dgp_cfg = dict(config.get("dgp", {}))
    dgp_name = str(dgp_cfg.get("name", "linear")).lower()
    runtime_cfg = dict(config.get("runtime", {}))
    ensemble_cfg = dict(config.get("ot_rank_ensemble", {}))
    scheduler_cfg = dict(config.get("resource_scheduler", {}))
    monitor_cfg = dict(config.get("system_monitor", runtime_cfg.get("system_monitor", {})))
    runtime_device = str(runtime_cfg.get("device", "cpu"))
    default_start_method = "spawn" if runtime_device.startswith("cuda") else "fork"
    multiprocessing_start_method = str(
        runtime_cfg.get("multiprocessing_start_method", default_start_method)
    )
    dependency_start_method = str(
        runtime_cfg.get("dependency_start_method", default_start_method)
    )
    if dgp_name in {"gaussian_nonlinear", "nonlinear", "relu"}:
        noises = deepcopy(config.get("noises", NOISES))
        if noise_labels:
            requested_noise_labels = {str(label) for label in noise_labels}
            noises = [noise for noise in noises if str(noise.get("label", noise.get("name", ""))) in requested_noise_labels]
        data_extra = {"dgp": "gaussian_nonlinear", "x_center": False}
    else:
        noises = deepcopy(config.get("noises", NOISES))
        if noise_labels:
            requested_noise_labels = {str(label) for label in noise_labels}
            noises = [noise for noise in noises if str(noise.get("label", noise.get("name", ""))) in requested_noise_labels]
            missing = sorted(requested_noise_labels - {str(noise.get("label", noise.get("name", ""))) for noise in noises})
            if missing:
                raise ValueError(f"Requested noise labels are absent from config: {missing}")
        data_extra = {
            "dgp": "linear",
            "coef_style": str(dgp_cfg.get("coef_style", "gaussian")),
            "singular_decay": float(dgp_cfg.get("singular_decay", 0.7)),
            "mu_nonzero": int(dgp_cfg.get("mu_nonzero", 0)),
            "mu_value": float(dgp_cfg.get("mu_value", 0.0)),
            "x_center": bool(dgp_cfg.get("x_center", True)),
        }
    return {
        "allow_exploratory": bool(config.get("allow_exploratory", False)),
        "data": {
            "coef_rank": None,
            "d": int(d),
            "n_test": int(config.get("n_test", 10000)),
            "n_train": int(n_train),
            "noise_scale": 1.0,
            "p": int(p),
            "signal": 1.0,
            "x_rho": float(config.get("x_rho", 0.0)),
            **data_extra,
        },
        "methods": _methods(config),
        "noises": noises,
        "rep_indices": rep_indices,
        "rotation_diagnostic": bool(config.get("rotation_diagnostic", True)),
        "jacobian_diagnostic": bool(config.get("jacobian_diagnostic", True)),
        "relu_init_seed_by_rep": bool(config.get("relu_init_seed_by_rep", False)),
        "runtime": {
            "device": runtime_device,
            "dtype": str(runtime_cfg.get("dtype", "float32")),
            "multiprocessing_start_method": multiprocessing_start_method,
            "parallel_backend": "thread",
            "torch_num_threads": 1,
            "workers": int(workers),
            "dependency_scheduler": True,
            "dependency_backend": "process",
            "disable_worker_progress": True,
            "dependency_start_method": dependency_start_method,
            "resource_scheduler": bool(scheduler_cfg.get("enabled", False)),
            "resource_cpu_workers": int(scheduler_cfg.get("cpu_workers", workers)),
            "resource_gpu_workers": int(scheduler_cfg.get("gpu_workers", 1)),
            "resource_gpu_torch_threads": int(scheduler_cfg.get("gpu_torch_threads", 1)),
            "gpu_batch_reference_seeds": bool(scheduler_cfg.get("gpu_batch_reference_seeds", True)),
            "reference_flush_every": int(scheduler_cfg.get("reference_flush_every", 5)),
            "resource_cpu_ot_workers": 0,
            "resource_cpu_ot_threads": 1,
            "resource_cpu_ot_checkpoint_every": int(scheduler_cfg.get("cpu_ot_checkpoint_every", 5)),
            "allow_cpu_ot_fallback": False,
            "system_monitor": {
                "enabled": bool(monitor_cfg.get("enabled", True)),
                "interval_sec": float(monitor_cfg.get("interval_sec", 5)),
                "filename": str(monitor_cfg.get("filename", "system_monitor.csv")),
            },
            "reference_seed_start": int(ensemble_cfg.get("reference_seed_start", config.get("ot_rank", {}).get("reference_seed", 20260721))),
            "reference_ensemble_k": int(ensemble_cfg.get("reference_ensemble_k", ensemble_cfg.get("num_reference_seeds", 1))),
            "report_ensemble_k": [int(value) for value in ensemble_cfg.get("report_k", [ensemble_cfg.get("reference_ensemble_k", 1)])],
        },
        "seeds": seeds,
    }


def validate_paper_request(config: dict, *, sample_sizes: list[int], dimensions: list[dict], methods: list[str] | None) -> None:
    if bool(config.get("allow_exploratory", False)):
        method_keys = methods if methods is not None else list(config.get("methods", []))
        unknown = [name for name in method_keys if name not in PAPER_METHOD_KEYS]
        if unknown:
            raise ValueError(f"Unknown method keys: {unknown}")
        return

    dgp_name = str(dict(config.get("dgp", {})).get("name", "linear")).lower()
    if dgp_name in {"gaussian_nonlinear", "nonlinear", "relu"}:
        bad_dims = sorted({(int(dim["p"]), int(dim["d"])) for dim in dimensions} - {(8, 4)})
        if bad_dims:
            raise ValueError("Gaussian nonlinear ReLU simulations require dimension (8,4)")
        bad_sizes = sorted(set(int(n) for n in sample_sizes) - PAPER_SAMPLE_SIZES)
        if bad_sizes:
            raise ValueError(f"Only sample sizes {sorted(PAPER_SAMPLE_SIZES)} are supported")
        method_keys = methods if methods is not None else list(config.get("methods", []))
        unknown = [name for name in method_keys if name not in PAPER_METHOD_KEYS]
        if unknown:
            raise ValueError(f"Unknown method keys: {unknown}")
        return

    bad_sizes = sorted(set(int(n) for n in sample_sizes) - PAPER_SAMPLE_SIZES)
    if bad_sizes:
        raise ValueError(f"Only paper sample sizes are supported: {sorted(PAPER_SAMPLE_SIZES)}")
    bad_dims = sorted({(int(dim["p"]), int(dim["d"])) for dim in dimensions} - PAPER_DIMENSIONS)
    if bad_dims:
        raise ValueError(f"Only paper dimensions are supported: {sorted(PAPER_DIMENSIONS)}")
    noise_labels = [str(noise.get("label", "")) for noise in config.get("noises", NOISES)]
    if noise_labels != ["A", "B", "C"]:
        raise ValueError("Noise labels must be exactly A, B, C in paper order")
    method_keys = methods if methods is not None else list(config.get("methods", []))
    unknown = [name for name in method_keys if name not in PAPER_METHOD_KEYS]
    if unknown:
        raise ValueError(f"Only paper method keys are supported; unknown keys: {unknown}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", default=None, help="Run one named config stage; staged configs run all stages by default.")
    parser.add_argument("--out-root", type=Path, default=ROOT / "results")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--n-train", action="append", type=int, default=None)
    parser.add_argument("--dimension", action="append", default=None, help="Dimension as p,d. Can be repeated.")
    parser.add_argument("--method", action="append", default=None, help="Method key to include. Can be repeated.")
    parser.add_argument("--noise", action="append", default=None, help="Noise label to include. Can be repeated.")
    parser.add_argument("--rep-start", type=int, default=None, help="First Monte Carlo repetition index to include, 1-based.")
    parser.add_argument("--rep-end", type=int, default=None, help="Last Monte Carlo repetition index to include, inclusive.")
    parser.add_argument("--init-cache-dir", action="append", default=None, help="Existing result directory to search for pilot init fits.")
    parser.add_argument("--require-existing-init", action="store_true", help="Fail OT-rank jobs if the required pilot fit is absent.")
    parser.add_argument("--reps", type=int, default=None, help="Override number of Monte Carlo repetitions.")
    parser.add_argument("--hardware-profile", default=None, help="Hardware tile profile, e.g. rtx3090 or a100.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    if args.reps is not None and not args.dry_run:
        raise ValueError("--reps is only available with --dry-run; paper runs use 50 repetitions.")
    if args.reps is not None:
        config["reps"] = int(args.reps)
    sample_sizes = args.n_train or [int(v) for v in config["sample_sizes"]]
    if args.dimension:
        dimensions = [{"p": int(v.split(",")[0]), "d": int(v.split(",")[1])} for v in args.dimension]
    else:
        dimensions = list(config["dimensions"])
    cw_cqr_section = dict(config.get("cw_cqr", {}))
    cw_cqr_method = str(cw_cqr_section.get("method_id", "")) or None
    available_methods = list(config.get("paper_methods", config.get("methods", [])))
    if args.method:
        unknown = sorted(set(args.method) - set(available_methods))
        if unknown:
            raise ValueError(f"Requested methods are absent from the paper config: {unknown}")
    cw_cqr_stage_requested = bool(cw_cqr_method and args.stage == "cw_cqr")
    run_cw_cqr = bool(
        cw_cqr_method
        and (args.stage is None or cw_cqr_stage_requested)
        and (not args.method or cw_cqr_method in args.method)
    )
    main_requested = (
        [method for method in args.method if method != cw_cqr_method]
        if args.method
        else None
    )
    staged = (
        _stage_configs(
            config,
            requested_stage=None if cw_cqr_stage_requested else args.stage,
            requested_methods=main_requested,
        )
        if not cw_cqr_stage_requested and (main_requested is None or main_requested)
        else []
    )

    generated_root = ROOT / "_generated_configs"
    generated_root.mkdir(parents=True, exist_ok=True)
    runner = ROOT / "src" / "otrank" / "experiments" / "run_simulation.py"
    for stage_name, stage_config in staged:
        validate_paper_request(
            stage_config,
            sample_sizes=sample_sizes,
            dimensions=dimensions,
            methods=list(stage_config.get("methods", [])),
        )
        workers = int(args.workers or stage_config.get("workers", 8))
        experiment_name = _safe_name(stage_config.get("experiment", "simulation"))
        if stage_name is not None:
            print(f"# stage={stage_name}")
        for dim in dimensions:
            for n_train in sample_sizes:
                low = build_low_level_config(
                    stage_config,
                    n_train=int(n_train),
                    p=int(dim["p"]),
                    d=int(dim["d"]),
                    workers=workers,
                    hardware_profile=args.hardware_profile or stage_config.get("hardware_profile"),
                    noise_labels=args.noise,
                    rep_start=args.rep_start,
                    rep_end=args.rep_end,
                    init_cache_dirs=args.init_cache_dir,
                    require_existing_init=args.require_existing_init or None,
                )
                cfg_path = generated_root / f"{experiment_name}_n{n_train}_p{dim['p']}_d{dim['d']}.json"
                cfg_path.write_text(json.dumps(low, indent=2, sort_keys=True))
                out = args.out_root / "main" if cw_cqr_section else args.out_root
                cmd = [sys.executable, str(runner), "--config", str(cfg_path), "--out", str(out), "--workers", str(workers)]
                print(" ".join(cmd))
                if not args.dry_run:
                    subprocess.run(cmd, check=True)

    if run_cw_cqr:
        if len(sample_sizes) != 1 or len(dimensions) != 1:
            raise ValueError("CW-CQR paper stage requires one sample size and one dimension")
        n_train = int(sample_sizes[0])
        p = int(dimensions[0]["p"])
        d = int(dimensions[0]["d"])
        protocol_root = args.out_root / "baselines"
        protocol_root.mkdir(parents=True, exist_ok=True)
        protocol_path = protocol_root / "simulation_protocol.json"
        protocol_path.write_text(
            json.dumps(
                build_cw_cqr_protocol(config, n_train=n_train, p=p, d=d),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        cqr_command = [
            sys.executable,
            "-m",
            "otrank.experiments.run_cw_cqr_simulation",
            "--config",
            str(args.config),
            "--handoff",
            str(protocol_root),
            "--out",
            str(args.out_root / "cw_cqr"),
            "--rep-start",
            str(args.rep_start or 1),
            "--rep-end",
            str(args.rep_end or int(config["reps"])),
            "--workers",
            str(int(args.workers or cw_cqr_section.get("execution", {}).get("workers", 1))),
        ]
        for noise in args.noise or []:
            cqr_command.extend(["--noise", str(noise)])
        print("# stage=cw_cqr")
        print("PYTHONPATH=src " + " ".join(cqr_command))
        if not args.dry_run:
            environment = dict(os.environ)
            source = str(ROOT / "src")
            environment["PYTHONPATH"] = source + (
                os.pathsep + environment["PYTHONPATH"]
                if environment.get("PYTHONPATH")
                else ""
            )
            subprocess.run(cqr_command, check=True, env=environment)


if __name__ == "__main__":
    main()
