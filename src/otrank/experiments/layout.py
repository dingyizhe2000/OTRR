"""Canonical result layout, checkpoint I/O, and metric summaries."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

import torch

from otrank.metrics import principal_angles

ROOT = Path(__file__).resolve().parents[3]
_SUMMARY_LOCK = threading.Lock()

def _safe_component(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)
    return text.strip("_") or "unnamed"


def _noise_label(noise: dict[str, Any]) -> str:
    return str(noise.get("label", noise["name"]))


def _method_label(method: dict[str, Any]) -> str:
    return str(method.get("save_label", method.get("label", method["name"])))


_CANONICAL_METHOD_ALIASES = {
    "ot_rank_cw_asm_init": "ot_rank_asm_init",
}


def _canonical_method_label(method: dict[str, Any]) -> str:
    """Return the stable method directory name without reference-seed leakage."""

    explicit = method.get("_canonical_base_label")
    label = str(explicit) if explicit is not None else _method_label(method)
    label = re.sub(r"_refseed\d+$", "", label)
    return _CANONICAL_METHOD_ALIASES.get(label, label)


def _is_iteration_method(method_name: str) -> bool:
    return str(method_name) in {"ot_rank_pnw_fullrisk", "relu_pnw_fullrisk"}


def _distribution_dir_name(data_cfg: dict[str, Any], noise: dict[str, Any]) -> str:
    dgp = str(data_cfg.get("dgp", "linear"))
    model_family = "relu" if dgp == "gaussian_nonlinear" else dgp
    return "_".join(
        _safe_component(part)
        for part in [
            model_family,
            f"n{data_cfg['n_train']}",
            f"p{data_cfg['p']}",
            f"d{data_cfg['d']}",
            f"noise{_noise_label(noise)}",
        ]
    )


def _rep_dir(
    out_dir: Path,
    data_cfg: dict[str, Any],
    noise: dict[str, Any],
    method: dict[str, Any],
    rep_index: int,
) -> Path:
    rep_dir = (
        out_dir
        / _distribution_dir_name(data_cfg, noise)
        / _safe_component(_canonical_method_label(method))
        / f"rep{rep_index:03d}"
    )
    reference_index = method.get("_reference_index")
    return rep_dir / f"K{int(reference_index):03d}" if reference_index is not None else rep_dir


def _ensemble_state_path(
    out_dir: Path,
    data_cfg: dict[str, Any],
    noise: dict[str, Any],
    method: dict[str, Any],
    rep_index: int,
) -> Path:
    return _rep_dir(out_dir, data_cfg, noise, method, rep_index) / "ensemble_state.pt"


def _ensemble_metrics_path(
    out_dir: Path,
    data_cfg: dict[str, Any],
    noise: dict[str, Any],
    method: dict[str, Any],
    rep_index: int,
) -> Path:
    return _rep_dir(out_dir, data_cfg, noise, method, rep_index) / "ensemble_metrics.json"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp_name)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _load_completed_job_rows(
    fit_save_dir: Path,
    method_name: str,
    method_params: dict[str, Any],
) -> list[dict[str, Any]] | None:
    """Return existing completed metric rows for a job, if it is complete."""

    if _is_iteration_method(method_name):
        eval_iters = [int(value) for value in method_params.get("eval_iters", [])]
        if not eval_iters:
            metrics_path = fit_save_dir / "metrics.json"
            if not metrics_path.exists():
                return None
            return [json.loads(metrics_path.read_text())]

        required = [fit_save_dir / f"metrics_iter{iteration:03d}.json" for iteration in eval_iters]
        if any(not path.exists() for path in required):
            return None

        rows: list[dict[str, Any]] = []
        for iteration in eval_iters:
            rows.append(json.loads((fit_save_dir / f"metrics_iter{iteration:03d}.json").read_text()))
        return rows

    metrics_path = fit_save_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    return [json.loads(metrics_path.read_text())]


def _save_fit(path: Path, fit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(fit, "state_dict_for_save"):
        state = fit.state_dict_for_save()
    elif hasattr(fit, "coef"):
        state = {
            "coef": fit.coef.detach().cpu(),
            "x_mean": fit.x_mean.detach().cpu(),
            "y_mean": fit.y_mean.detach().cpu(),
            "method": fit.method,
        }
    else:
        raise TypeError(f"Do not know how to serialize fit of type {type(fit)!r}")
    torch.save(state, path)


def _candidate_init_labels(method_params: dict[str, Any]) -> list[str]:
    label = method_params.get("init_fit_label")
    if label is None:
        labels = ["zero_intercept_ols"]
    elif isinstance(label, list):
        labels = [str(item) for item in label]
    else:
        labels = [str(label)]
    return labels


def _candidate_init_out_dirs(out_dir: Path, method_params: dict[str, Any]) -> list[Path]:
    dirs = [out_dir]
    for value in method_params.get("init_cache_dirs", []) or []:
        path = Path(str(value)).expanduser()
        dirs.append(path if path.is_absolute() else (ROOT / path))
    return dirs


def _init_fit_path_for_label(
    out_dir: Path,
    data_cfg: dict[str, Any],
    noise: dict[str, Any],
    rep_index: int,
    label: str,
) -> Path:
    return (
        _rep_dir(
            out_dir,
            data_cfg,
            noise,
            {"name": label, "label": label, "save_label": label},
            rep_index,
        )
        / "fit.pt"
    )


def _find_init_fit_path(
    out_dir: Path,
    data_cfg: dict[str, Any],
    noise: dict[str, Any],
    rep_index: int,
    method_params: dict[str, Any],
) -> Path | None:
    for search_dir in _candidate_init_out_dirs(out_dir, method_params):
        for label in _candidate_init_labels(method_params):
            candidates = [
                _init_fit_path_for_label(search_dir, data_cfg, noise, rep_index, label),
                search_dir
                / _distribution_dir_name(data_cfg, noise).replace(f"noise{_noise_label(noise)}", _noise_label(noise))
                / _safe_component(label)
                / f"{rep_index:03d}"
                / "fit.pt",
            ]
            for path in candidates:
                if path.exists():
                    return path
    return None


def _require_existing_init(method_params: dict[str, Any]) -> bool:
    return bool(method_params.get("require_existing_init", False))


def _rotated_init_fit_path(
    *,
    source_path: Path,
    target_path: Path,
    q: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Path | None:
    if not source_path.exists():
        return None
    state = _load_checkpoint(source_path, device)
    coef = state.get("coef")
    if coef is None:
        return None
    coef = coef.to(device=device, dtype=dtype)
    rotated_state = dict(state)
    rotated_state["coef"] = (q @ coef).detach().cpu()
    if "y_mean" in rotated_state:
        y_mean = rotated_state["y_mean"].to(device=device, dtype=dtype)
        rotated_state["y_mean"] = (y_mean @ q.T).detach().cpu()
    if "x_mean" in rotated_state:
        rotated_state["x_mean"] = rotated_state["x_mean"].to(device=device, dtype=dtype).detach().cpu()
    rotated_state["method"] = f"rotated_init_from_{state.get('method', source_path.parent.name)}"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rotated_state, target_path)
    return target_path


def _principal_angle_stats(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    angles = principal_angles(left, right)
    if angles.numel() == 0:
        return float("nan"), float("nan")
    return float(torch.max(angles).item()), float(torch.linalg.norm(torch.sin(angles)).item())


def _rowspace_principal_angle_stats(left_coef: torch.Tensor, right_coef: torch.Tensor) -> tuple[float, float]:
    return _principal_angle_stats(left_coef.T, right_coef.T)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _collect_metric_rows(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.rglob("*metrics*.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict) and "noise" in payload and "method" in payload:
            rows.append(payload)
        elif isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
            for by_iteration in payload["metrics"].values():
                if not isinstance(by_iteration, dict):
                    continue
                for row in by_iteration.values():
                    if isinstance(row, dict) and "noise" in row and "method" in row:
                        rows.append(row)
    return rows


def _write_summary_from_rows(rows: list[dict[str, Any]], out_path: Path) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["noise"], row["method"]), []).append(row)
    metrics = [
        "prediction_mse",
        "prediction_mse_per_dim",
        "prediction_rmse_per_dim",
        "prediction_mae",
        "translation_adjusted_mse",
        "centered_prediction_mse",
        "mean_bias",
        "coef_error",
        "e_max",
        "coef_angle",
        "jacobian_estimation_rmse_per_entry",
        "rotation_pred_gap",
        "rotation_pred_mse_per_dim",
        "rotation_pred_rmse_per_dim",
        "rotation_pred_angle",
        "rotation_principal_angle_max",
        "rotation_principal_angle_fro_sin",
        "rotation_coef_principal_angle_max",
        "rotation_coef_principal_angle_fro_sin",
        "rotation_coef_weighted_sine_discrepancy",
        "rotation_response_principal_angle_max",
        "rotation_response_principal_angle_fro_sin",
        "rotation_jacobian_rmse_per_entry",
        "rotation_jacobian_relative_gap",
        "rotation_jacobian_rowspace_max_angle_mean",
        "rotation_jacobian_rowspace_max_angle_median",
        "rotation_jacobian_rowspace_max_angle_q90",
        "rotation_jacobian_rowspace_fro_sin_mean",
        "rotation_jacobian_projection_distance",
        "rotation_jacobian_full_rank_fraction",
        "rotation_jacobian_rank_mismatch_fraction",
        "runtime_sec",
    ]
    summary: list[dict[str, Any]] = []
    for (noise, method), group in sorted(groups.items()):
        out_row: dict[str, Any] = {"noise": noise, "method": method, "n_reps": len(group)}
        for metric in metrics:
            values = [float(r[metric]) for r in group if metric in r and math.isfinite(float(r[metric]))]
            if values:
                mean = sum(values) / len(values)
                out_row[f"{metric}_mean"] = mean
                if len(values) > 1:
                    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                    out_row[f"{metric}_sd"] = var**0.5
                else:
                    out_row[f"{metric}_sd"] = 0.0
        summary.append(out_row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    csv_path = out_path.with_suffix(".csv")
    fieldnames = sorted({key for row in summary for key in row})
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def _refresh_partial_summary(out_dir: Path) -> None:
    with _SUMMARY_LOCK:
        rows = _collect_metric_rows(out_dir)
        if rows:
            _write_summary_from_rows(rows, out_dir / "partial_summary.json")
