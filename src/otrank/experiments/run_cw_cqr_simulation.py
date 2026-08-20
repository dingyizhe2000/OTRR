"""Run the paper's n=1000 A/B/C simulations for the CW-CQR baseline."""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from otrank.data.noise import make_generator, random_orthogonal
from otrank.data.sim import make_linear_simulation
from otrank.estimators.cw_cqr import cw_cqr_covariance
from otrank.estimators.cw_cqr_bridge import (
    CW_CQR_METHOD_ID,
    CWCQRFitConfig,
    equally_spaced_quantiles,
    fit_cw_cqr,
)
from otrank.metrics import principal_angles, weighted_sine_discrepancy
from otrank.vectorization import (
    COEFFICIENT_VECTORIZATION,
    require_vectorization,
    vec_coefficient,
)


DEFAULT_HANDOFF = Path("results/simulation/linear_n1000_p20_d10_final/source/baselines")
DEFAULT_OUT = Path("results/simulation/linear_n1000_p20_d10_final/source/cw_cqr")
DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "simulation_linear.json"


def _noise_specs(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(spec["label"]): spec
        for spec in protocol["formal_config"]["noises"]
    }


def _simulation(
    protocol: dict[str, Any], noise: dict[str, Any], seed: int
):
    target = protocol["target"]
    dgp = protocol["coefficient_generation"]
    return make_linear_simulation(
        n_train=int(target["n_train"]),
        n_test=int(protocol["formal_config"]["n_test"]),
        p=int(target["p"]),
        d=int(target["d"]),
        noise=noise,
        seed=int(seed),
        x_rho=float(protocol["formal_config"].get("x_rho", 0.0)),
        x_center=bool(dgp["x_center"]),
        mu_nonzero=int(dgp["mu_nonzero"]),
        mu_value=float(dgp["mu_value"]),
        coef_style=str(dgp["coef_style"]),
        singular_decay=float(dgp["singular_decay"]),
        device="cpu",
        dtype=torch.float32,
    )


def _rotation(dimension: int, seed: int) -> np.ndarray:
    generator = make_generator(seed + 10_000, device="cpu")
    q = random_orthogonal(
        dimension,
        generator,
        device="cpu",
        dtype=torch.float32,
    )
    return q.numpy().astype(np.float64)


def _rowspace_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_tensor = torch.as_tensor(left, dtype=torch.float64)
    right_tensor = torch.as_tensor(right, dtype=torch.float64)
    angles = principal_angles(left_tensor.T, right_tensor.T)
    if angles.numel() == 0:
        raise ValueError("CW-CQR coefficient row space has numerical rank zero")
    weighted, weighted_angle = weighted_sine_discrepancy(
        left_tensor,
        right_tensor,
        return_angle_degrees=True,
    )
    angle_degrees = torch.rad2deg(angles)
    return {
        "rotation_coef_principal_angle_max_deg": float(angle_degrees.max()),
        "rotation_coef_principal_angle_mean_deg": float(angle_degrees.mean()),
        "rotation_coef_principal_angle_rms_deg": float(
            torch.sqrt(torch.mean(angle_degrees.square()))
        ),
        "rotation_coef_principal_angle_fro_sin": float(
            torch.linalg.vector_norm(torch.sin(angles))
        ),
        "rotation_coef_weighted_sine_discrepancy": float(weighted),
        "rotation_coef_weighted_sine_angle_deg": float(weighted_angle),
        "rotation_coef_relative_fro_gap": float(
            np.linalg.norm(left - right) / np.linalg.norm(left)
        ),
    }


def _complete_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required = {
        "noise",
        "rep",
        "coef_error",
        "coverage_90",
        "coverage_95",
        "rotation_coef_principal_angle_max_deg",
        "vectorization",
    }
    if not required.issubset(payload):
        return None
    try:
        require_vectorization(payload)
    except ValueError:
        return None
    numeric = [
        value
        for key, value in payload.items()
        if key in required - {"noise", "vectorization"}
    ]
    if not all(np.isfinite(float(value)) for value in numeric):
        return None
    return payload


def _run_rep(job: dict[str, Any]) -> dict[str, Any]:
    protocol = job["protocol"]
    noise_label = str(job["noise"])
    rep = int(job["rep"])
    seed = int(protocol["seed_convention"]["seed_start"]) + rep - 1
    out = Path(job["out"]) / f"noise_{noise_label}" / f"rep{rep:03d}"
    metrics_path = out / "metrics.json"
    completed = _complete_metrics(metrics_path)
    if completed is not None:
        return completed

    out.mkdir(parents=True, exist_ok=True)
    noise = _noise_specs(protocol)[noise_label]
    simulation = _simulation(protocol, noise, seed)
    x_train = simulation.x_train.numpy().astype(np.float64)
    y_train = simulation.y_train.numpy().astype(np.float64)
    x_test = simulation.x_test.numpy().astype(np.float64)
    truth_prediction = simulation.f_test.numpy().astype(np.float64)
    truth_coef = simulation.coef.numpy().astype(np.float64)
    tau = equally_spaced_quantiles(int(job["quantile_levels"]))
    fit_config = CWCQRFitConfig(
        tau=tuple(tau),
        method=str(job["method"]),
        max_iterations=int(job["max_iterations"]),
        tolerance=float(job["tolerance"]),
        rscript=str(job["rscript"]),
        r_library=job["r_lib"],
    )

    original_fit = fit_cw_cqr(
        x_train,
        y_train,
        out / "original",
        config=fit_config,
        overwrite=bool(job["overwrite_fits"]),
    )
    beta = original_fit.beta
    intercepts = original_fit.quantile_intercepts
    diagnostics = original_fit.diagnostics

    q = _rotation(y_train.shape[1], seed)
    rotated_fit = fit_cw_cqr(
        x_train,
        y_train @ q.T,
        out / "rotated",
        config=fit_config,
        overwrite=bool(job["overwrite_fits"]),
    )
    beta_rotated = rotated_fit.beta
    rotated_diagnostics = rotated_fit.diagnostics
    beta_rotated_back = q.T @ beta_rotated

    inference = cw_cqr_covariance(
        x_train,
        y_train,
        beta,
        intercepts,
        tau,
        derivative_ridge=float(job["derivative_ridge"]),
    )
    n, d = y_train.shape
    p = x_train.shape[1]
    flat_error = np.asarray(vec_coefficient(beta - truth_coef), dtype=np.float64)
    marginal_variance = np.clip(np.diag(inference.covariance), 0.0, np.inf)
    marginal_se = np.sqrt(marginal_variance / n)
    half_width_90 = float(norm.ppf(0.95)) * marginal_se
    half_width_95 = float(norm.ppf(0.975)) * marginal_se
    coverage_90 = np.abs(flat_error) <= half_width_90
    coverage_95 = np.abs(flat_error) <= half_width_95

    prediction = x_test @ beta.T
    prediction_error = prediction - truth_prediction
    centered_error = prediction_error - prediction_error.mean(axis=0, keepdims=True)
    angles = _rowspace_metrics(beta, beta_rotated_back)

    np.savez_compressed(
        out / "fit_and_inference.npz",
        beta=beta,
        quantile_intercepts=intercepts,
        beta_rotated=beta_rotated,
        rotation=q,
        beta_rotated_back=beta_rotated_back,
        truth_coef=truth_coef,
        covariance=inference.covariance,
        influence=inference.influence,
        density_at_quantiles=inference.density_at_quantiles,
        marginal_se=marginal_se,
        coverage_90=coverage_90,
        coverage_95=coverage_95,
        tau=tau,
        vectorization=np.asarray(COEFFICIENT_VECTORIZATION),
    )
    payload: dict[str, Any] = {
        "noise": noise_label,
        "rep": rep,
        "seed": seed,
        "n_train": n,
        "n_test": len(x_test),
        "p": p,
        "d": d,
        "vectorization": COEFFICIENT_VECTORIZATION,
        "quantile_levels": len(tau),
        "coef_error": float(np.linalg.norm(flat_error)),
        "prediction_mse": float(np.mean(np.sum(np.square(prediction_error), axis=1))),
        "prediction_mse_per_dim": float(np.mean(np.square(prediction_error))),
        "prediction_rmse_per_dim": float(np.sqrt(np.mean(np.square(prediction_error)))),
        "prediction_mae": float(np.mean(np.abs(prediction_error))),
        "centered_prediction_mse": float(
            np.mean(np.sum(np.square(centered_error), axis=1))
        ),
        "coverage_90": float(np.mean(coverage_90)),
        "coverage_95": float(np.mean(coverage_95)),
        "average_half_width_90": float(np.mean(half_width_90)),
        "average_half_width_95": float(np.mean(half_width_95)),
        "average_marginal_se": float(np.mean(marginal_se)),
        "covariance_rank": int(np.linalg.matrix_rank(inference.covariance)),
        "bread_condition_max": float(np.max(inference.bread_condition_numbers)),
        "bread_condition_median": float(np.median(inference.bread_condition_numbers)),
        "fit_runtime_sec": float(diagnostics["runtime_sec"].sum()),
        "rotated_fit_runtime_sec": float(rotated_diagnostics["runtime_sec"].sum()),
        "fit_wall_sec": float(original_fit.wall_sec),
        "rotated_fit_wall_sec": float(rotated_fit.wall_sec),
        "fit_resumed": bool(original_fit.resumed),
        "rotated_fit_resumed": bool(rotated_fit.resumed),
        **angles,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _mean_sd(values: pd.Series) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1))


def _summarize(out: Path, target_reps: int) -> pd.DataFrame:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(out.glob("noise_*/rep*/metrics.json"))
    ]
    metrics = pd.DataFrame(rows).sort_values(["noise", "rep"])
    metrics.to_csv(out / "repetition_metrics.csv", index=False)
    summaries: list[dict[str, Any]] = []
    for noise, group in metrics.groupby("noise", sort=True):
        if len(group) != target_reps:
            continue
        coverage90_mean, coverage90_sd = _mean_sd(group["coverage_90"])
        coverage95_mean, coverage95_sd = _mean_sd(group["coverage_95"])
        summary: dict[str, Any] = {
            "noise": noise,
            "reps": len(group),
            "coverage_90_mean": coverage90_mean,
            "coverage_90_sd": coverage90_sd,
            "coverage_90_mcse": coverage90_sd / math.sqrt(len(group)),
            "coverage_95_mean": coverage95_mean,
            "coverage_95_sd": coverage95_sd,
            "coverage_95_mcse": coverage95_sd / math.sqrt(len(group)),
        }
        if len(group) < 2:
            summary.update(
                {
                    "median_marginal_se_mc_ratio": float("nan"),
                    "iqr_marginal_se_mc_ratio": float("nan"),
                }
            )
            for column in (
                "coef_error",
                "prediction_rmse_per_dim",
                "prediction_mae",
                "centered_prediction_mse",
                "rotation_coef_principal_angle_max_deg",
                "rotation_coef_principal_angle_fro_sin",
                "rotation_coef_weighted_sine_discrepancy",
                "average_half_width_90",
                "average_half_width_95",
                "fit_runtime_sec",
                "rotated_fit_runtime_sec",
            ):
                summary[f"{column}_mean"] = float(group[column].mean())
                summary[f"{column}_sd"] = float("nan")
            summaries.append(summary)
            continue
        errors: list[np.ndarray] = []
        structured_diagonals: list[np.ndarray] = []
        for rep in group["rep"]:
            state_path = out / f"noise_{noise}" / f"rep{int(rep):03d}" / "fit_and_inference.npz"
            with np.load(state_path, allow_pickle=False) as state:
                require_vectorization(state)
                errors.append(
                    np.asarray(
                        vec_coefficient(state["beta"] - state["truth_coef"]),
                        dtype=np.float64,
                    )
                )
                structured_diagonals.append(np.diag(state["covariance"]))
        error_matrix = np.stack(errors)
        centered = error_matrix - error_matrix.mean(axis=0, keepdims=True)
        n = int(group["n_train"].iloc[0])
        mc_covariance = n * centered.T @ centered / (len(group) - 1)
        np.savez_compressed(
            out / f"mc_covariance_noise_{noise}.npz",
            covariance=mc_covariance,
            vectorization=np.asarray(COEFFICIENT_VECTORIZATION),
        )
        structured_diagonal = np.mean(np.stack(structured_diagonals), axis=0)
        mc_diagonal = np.diag(mc_covariance)
        valid = mc_diagonal > 0
        se_ratio = np.sqrt(structured_diagonal[valid] / mc_diagonal[valid])
        summary.update({
            "median_marginal_se_mc_ratio": float(np.median(se_ratio)),
            "iqr_marginal_se_mc_ratio": float(
                np.quantile(se_ratio, 0.75) - np.quantile(se_ratio, 0.25)
            ),
        })
        for column in (
            "coef_error",
            "prediction_rmse_per_dim",
            "prediction_mae",
            "centered_prediction_mse",
            "rotation_coef_principal_angle_max_deg",
            "rotation_coef_principal_angle_fro_sin",
            "rotation_coef_weighted_sine_discrepancy",
            "average_half_width_90",
            "average_half_width_95",
            "fit_runtime_sec",
            "rotated_fit_runtime_sec",
        ):
            summary[f"{column}_mean"] = float(group[column].mean())
            summary[f"{column}_sd"] = float(group[column].std(ddof=1))
        summaries.append(summary)
    result = pd.DataFrame(summaries)
    result.to_csv(out / "summary_by_noise.csv", index=False)
    return result


def _write_report(out: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# CW-CQR n=1000 simulation supplement",
        "",
        "CW-CQR uses 20 levels `tau=l/21`, one common slope and 20 nuisance",
        "quantile intercepts per response. The slope sandwich profiles the",
        "intercepts and stacks observation-level IFs across responses.",
        "",
        "`V` is the covariance of `sqrt(n) vec(Bhat-B0)`; estimator covariance",
        "is `V/n`. Rotation fits use the formal `seed+10000` orthogonal matrix",
        "and are aligned back before coefficient row-space diagnostics.",
        "",
    ]
    if summary.empty:
        lines.append("No noise setting has all requested repetitions yet.")
    else:
        lines.extend(
            [
                "## Complete settings",
                "",
                "```csv",
                summary.to_csv(index=False).strip(),
                "```",
            ]
        )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.handoff / "simulation_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    noises = args.noise or ["A", "B", "C"]
    unknown = sorted(set(noises) - set(_noise_specs(protocol)))
    if unknown:
        raise ValueError(f"Unknown noise labels: {unknown}")
    jobs = [
        {
            "protocol": protocol,
            "noise": noise,
            "rep": rep,
            "out": str(args.out.resolve()),
            "quantile_levels": args.quantile_levels,
            "method": args.method,
            "max_iterations": args.max_iterations,
            "tolerance": args.tolerance,
            "derivative_ridge": args.derivative_ridge,
            "rscript": args.rscript,
            "r_lib": args.r_lib,
            "overwrite_fits": args.overwrite_fits,
        }
        for noise in noises
        for rep in range(args.rep_start, args.rep_end + 1)
    ]
    config = {
        "paper_config_path": str(args.config.resolve()),
        "protocol_path": str(protocol_path.resolve()),
        "protocol": protocol,
        "noises": noises,
        "rep_start": args.rep_start,
        "rep_end": args.rep_end,
        "quantile_levels": args.quantile_levels,
        "tau": (np.arange(1, args.quantile_levels + 1) / (args.quantile_levels + 1)).tolist(),
        "cqr_method": args.method,
        "method_id": CW_CQR_METHOD_ID,
        "max_iterations": args.max_iterations,
        "tolerance": args.tolerance,
        "derivative_ridge": args.derivative_ridge,
        "workers": args.workers,
        "covariance_convention": "V=Cov(sqrt(n) vec(Bhat-B0)); Cov(Bhat)=V/n",
        "rotation_convention": "Q uses DGP seed+10000; rotated fit is aligned by Q.T",
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    started = time.perf_counter()
    # Each fit runs in an independent R subprocess, so threads provide process-
    # level CPU parallelism without relying on macOS Python semaphores.
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(_run_rep, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            print(
                f"done noise={result['noise']} rep={result['rep']:03d} "
                f"coef={result['coef_error']:.4f} "
                f"angle={result['rotation_coef_principal_angle_max_deg']:.3f} "
                f"wall={result['fit_wall_sec'] + result['rotated_fit_wall_sec']:.1f}s",
                flush=True,
            )
    summary = _summarize(args.out, args.rep_end - args.rep_start + 1)
    _write_report(args.out, summary)
    (args.out / "runtime.json").write_text(
        json.dumps({"total_wall_sec": time.perf_counter() - started}, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    known, _ = config_parser.parse_known_args()
    paper_config = json.loads(known.config.read_text(encoding="utf-8"))
    cw_cqr = dict(paper_config.get("cw_cqr", {}))
    fit = dict(cw_cqr.get("fit", {}))
    inference = dict(cw_cqr.get("inference", {}))
    execution = dict(cw_cqr.get("execution", {}))
    if not cw_cqr:
        raise ValueError(f"{known.config} has no cw_cqr section")

    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--noise", action="append", choices=("A", "B", "C"), default=[])
    parser.add_argument("--rep-start", type=int, default=1)
    parser.add_argument("--rep-end", type=int, default=50)
    parser.add_argument("--quantile-levels", type=int, default=int(cw_cqr["quantile_levels"]))
    parser.add_argument("--method", choices=("ip", "mm", "admm", "cd"), default=str(fit["method"]))
    parser.add_argument("--max-iterations", type=int, default=int(fit["max_iterations"]))
    parser.add_argument("--tolerance", type=float, default=float(fit["tolerance"]))
    parser.add_argument("--workers", type=int, default=int(execution["workers"]))
    parser.add_argument("--derivative-ridge", type=float, default=float(inference["derivative_ridge"]))
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument(
        "--r-lib",
        help="Optional R library path; otherwise use CQR_R_LIB/R_LIBS_USER/.libPaths()",
    )
    parser.add_argument("--overwrite-fits", action="store_true")
    args = parser.parse_args()
    if args.rep_start < 1 or args.rep_end < args.rep_start:
        parser.error("require 1 <= rep-start <= rep-end")
    if args.quantile_levels < 2:
        parser.error("quantile-levels must be at least 2")
    return args


if __name__ == "__main__":
    run(parse_args())
