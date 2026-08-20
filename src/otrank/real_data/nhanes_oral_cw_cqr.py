"""Five-fold CW-CQR prediction, invariance, and raw-B inference for NHANES."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from otrank.estimators.cw_cqr_bridge import (
    CW_CQR_METHOD_ID,
    CWCQRFitConfig,
    equally_spaced_quantiles,
    fit_cw_cqr,
    load_cw_cqr_fit,
)
from otrank.estimators.cw_cqr import cqr_significance_summary, cw_cqr_covariance
from otrank.metrics import principal_angles, weighted_sine_discrepancy
from otrank.real_data.nhanes_oral_baselines import DEFAULT_DATA_ROOT
from otrank.real_data.nhanes_cw_cqr_protocol import (
    P43_COLUMNS,
    load_prepared_cw_cqr_jobs,
    prepare_raw_cw_cqr_jobs,
    response_rotation,
)
from otrank.vectorization import COEFFICIENT_VECTORIZATION


def _coefficient_angles(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_t = torch.as_tensor(left, dtype=torch.float64)
    right_t = torch.as_tensor(right, dtype=torch.float64)
    angles = principal_angles(left_t.T, right_t.T)
    weighted, weighted_angle = weighted_sine_discrepancy(
        left_t, right_t, return_angle_degrees=True
    )
    return {
        "max_angle_deg": float(torch.rad2deg(angles.max()).item()),
        "mean_angle_deg": float(torch.rad2deg(angles).mean().item()),
        "rms_angle_deg": float(torch.sqrt(torch.mean(torch.rad2deg(angles) ** 2)).item()),
        "fro_sin": float(torch.linalg.vector_norm(torch.sin(angles)).item()),
        "weighted_sine_discrepancy": float(weighted),
        "weighted_sine_angle_deg": float(weighted_angle),
    }


def run(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    tau = equally_spaced_quantiles(int(args.quantile_levels))
    fit_config = CWCQRFitConfig(
        tau=tuple(tau),
        method=args.method,
        max_iterations=int(args.max_iterations),
        tolerance=float(args.tolerance),
        rscript=args.rscript,
        r_library=args.r_lib,
    )
    rotations = [int(value) for value in args.rotation_seed]
    if args.prepared_dir is not None:
        prepared, jobs, data_audit, n_total, predictor_names, response_names = (
            load_prepared_cw_cqr_jobs(
                args.prepared_dir,
                args.out,
                rotations,
                oof_target_file=args.oof_target_file,
            )
        )
    else:
        prepared, jobs, data_audit, n_total, predictor_names, response_names = (
            prepare_raw_cw_cqr_jobs(
                args.data_root,
                args.out,
                min_reads=int(args.min_reads),
                outer_folds=int(args.outer_folds),
                fold_limit=args.fold_limit,
                split_seed=int(args.seed),
                zero_method=args.zero_method,
                rotation_seeds=rotations,
            )
        )

    job_runtime_rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(
                fit_cw_cqr,
                job["x"],
                job["y"],
                job["out"],
                config=fit_config,
                overwrite=bool(args.overwrite_fits),
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            row = {
                "split": job["split"],
                "representation": job["label"],
                "wall_sec": result.wall_sec,
                "resumed": result.resumed,
            }
            job_runtime_rows.append(row)
            print(
                f"done fold={job['split']} representation={job['label']} "
                f"wall={result.wall_sec:.2f}s resumed={result.resumed}",
                flush=True,
            )

    metric_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    predictor_rows: list[dict[str, object]] = []
    invariance_rows: list[dict[str, object]] = []
    for split in sorted(prepared):
        fold = prepared[split]
        fold_dir = args.out / f"fold_{split:02d}"
        original_fit = load_cw_cqr_fit(fold_dir / "original")
        beta = original_fit.beta
        intercepts = original_fit.quantile_intercepts
        diagnostics = original_fit.diagnostics
        x_train = np.asarray(fold["x_train"])
        x_test = np.asarray(fold["x_test"])
        y_train = np.asarray(fold["y_train"])
        y_test = np.asarray(fold["y_test"])
        mean_intercept = np.mean(y_train - x_train @ beta.T, axis=0)
        prediction = mean_intercept[None, :] + x_test @ beta.T
        residual = y_test - prediction
        row_error = np.linalg.norm(residual, axis=1)
        baseline_sse = float(np.sum((y_test - y_train.mean(axis=0)) ** 2))
        sse = float(np.sum(residual**2))

        inference = cw_cqr_covariance(
            x_train,
            y_train,
            beta,
            intercepts,
            tau,
            derivative_ridge=float(args.derivative_ridge),
        )
        significance = cqr_significance_summary(beta, inference.covariance, len(x_train))
        np.savez_compressed(
            fold_dir / "cw_cqr_inference.npz",
            beta=beta,
            quantile_intercepts=intercepts,
            mean_intercept=mean_intercept,
            covariance=inference.covariance,
            influence=inference.influence,
            standard_errors=significance["standard_errors"],
            coefficient_p=significance["coefficient_p"],
            coefficient_q=significance["coefficient_q"],
            predictor_p=significance["predictor_p"],
            predictor_q=significance["predictor_q"],
            density_at_quantiles=inference.density_at_quantiles,
            vectorization=np.asarray(COEFFICIENT_VECTORIZATION),
        )
        metric_rows.append(
            {
                "split": split,
                "method": "CW-CQR",
                "n_train": len(x_train),
                "n_test": len(x_test),
                "predictor_dim": x_train.shape[1],
                "response_dim": y_train.shape[1],
                "vectorization": COEFFICIENT_VECTORIZATION,
                "quantile_levels": len(tau),
                "rmse_per_ilr_dim": float(np.sqrt(np.mean(residual**2))),
                "aitchison_rmse": float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
                "mae_per_ilr_dim": float(np.mean(np.abs(residual))),
                "test_r2": 1.0 - sse / baseline_sse,
                "row_error_median": float(np.median(row_error)),
                "row_error_q90": float(np.quantile(row_error, 0.90)),
                "row_error_q95": float(np.quantile(row_error, 0.95)),
                "row_error_q99": float(np.quantile(row_error, 0.99)),
                "row_error_max": float(np.max(row_error)),
                "avg_marginal_se": float(
                    np.mean(np.sqrt(np.clip(np.diag(inference.covariance), 0, np.inf) / len(x_train)))
                ),
                "covariance_rank": int(np.linalg.matrix_rank(inference.covariance)),
                "bread_condition_max": float(np.max(inference.bread_condition_numbers)),
                "bread_condition_median": float(np.median(inference.bread_condition_numbers)),
                "coef_significant_90": significance["coef_significant_90"],
                "coef_significant_95": significance["coef_significant_95"],
                "coef_bh_q10": significance["coef_bh_q10"],
                "coef_bh_q05": significance["coef_bh_q05"],
                "predictor_bh_q10": significance["predictor_bh_q10"],
                "predictor_bh_q05": significance["predictor_bh_q05"],
                "fit_runtime_sec": float(diagnostics["runtime_sec"].sum()),
            }
        )
        for response_index in range(beta.shape[0]):
            for predictor_index, predictor in enumerate(P43_COLUMNS):
                coefficient_rows.append(
                    {
                        "split": split,
                        "response_index": response_index + 1,
                        "predictor": predictor,
                        "coefficient": beta[response_index, predictor_index],
                        "se": significance["standard_errors"][response_index, predictor_index],
                        "p_value": significance["coefficient_p"][response_index, predictor_index],
                        "bh_q_value": significance["coefficient_q"][response_index, predictor_index],
                    }
                )
        for predictor_index, predictor in enumerate(P43_COLUMNS):
            predictor_rows.append(
                {
                    "split": split,
                    "predictor": predictor,
                    "p_value": significance["predictor_p"][predictor_index],
                    "bh_q_value": significance["predictor_q"][predictor_index],
                }
            )
        for seed in rotations:
            q = response_rotation(beta.shape[0], seed)
            beta_rotated = load_cw_cqr_fit(fold_dir / f"rotation_{seed}").beta
            aligned = q.T @ beta_rotated
            invariance_rows.append(
                {
                    "split": split,
                    "rotation_seed": seed,
                    **_coefficient_angles(beta, aligned),
                    "coef_relative_fro_gap": float(
                        np.linalg.norm(beta - aligned) / np.linalg.norm(beta)
                    ),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    invariance = pd.DataFrame(invariance_rows)
    metrics.to_csv(args.out / "fold_metrics.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(args.out / "coefficient_inference.csv", index=False)
    predictors_frame = pd.DataFrame(predictor_rows)
    predictors_frame.to_csv(args.out / "predictor_joint_wald.csv", index=False)
    pd.DataFrame(job_runtime_rows).sort_values(["split", "representation"]).to_csv(
        args.out / "runtime.csv", index=False
    )
    if not invariance.empty:
        invariance.to_csv(args.out / "invariance_metrics.csv", index=False)

    summary = {
        "folds": int(len(metrics)),
        "rmse_mean": float(metrics["rmse_per_ilr_dim"].mean()),
        "rmse_sd": float(metrics["rmse_per_ilr_dim"].std(ddof=1)),
        "mae_mean": float(metrics["mae_per_ilr_dim"].mean()),
        "r2_mean": float(metrics["test_r2"].mean()),
        "r2_sd": float(metrics["test_r2"].std(ddof=1)),
        "avg_marginal_se_mean": float(metrics["avg_marginal_se"].mean()),
        "predictor_bh_q05_mean": float(metrics["predictor_bh_q05"].mean()),
        "predictor_bh_q05_stable": int(
            (predictors_frame.assign(hit=predictors_frame["bh_q_value"] < 0.05)
             .groupby("predictor")["hit"].sum() >= max(1, math.ceil(0.8 * len(metrics)))).sum()
        ),
        "fit_runtime_sec_total": float(metrics["fit_runtime_sec"].sum()),
    }
    if not invariance.empty:
        for column in (
            "max_angle_deg",
            "fro_sin",
            "weighted_sine_discrepancy",
            "weighted_sine_angle_deg",
        ):
            summary[f"{column}_mean"] = float(invariance[column].mean())
            summary[f"{column}_max"] = float(invariance[column].max())
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    config = {
        "data_root": str(args.data_root),
        "prepared_dir": str(args.prepared_dir) if args.prepared_dir is not None else None,
        "oof_target_file": (
            str(args.oof_target_file) if args.oof_target_file is not None else None
        ),
        "n": int(n_total),
        "predictors": predictor_names,
        "response_components": response_names,
        "quantile_levels": tau.tolist(),
        "outer_folds": int(args.outer_folds),
        "split_seed": int(args.seed),
        "rotation_seeds": rotations,
        "zero_method": args.zero_method,
        "cqr_method": args.method,
        "method_id": CW_CQR_METHOD_ID,
        "max_iterations": int(args.max_iterations),
        "tolerance": float(args.tolerance),
        "workers": int(args.workers),
        "derivative_ridge": float(args.derivative_ridge),
        "covariance_convention": "V is covariance of sqrt(n) vec(Bhat); estimator covariance is V/n",
        "vectorization": COEFFICIENT_VECTORIZATION,
        "inference": "stacked coordinatewise CQR sandwich with quantile intercept nuisance",
        "data_audit": data_audit,
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        help="Authoritative fold_XX.npz directory; bypasses raw-data reconstruction",
    )
    parser.add_argument(
        "--oof-target-file",
        type=Path,
        help=(
            "Optional NPZ containing full-order y_true (and fold) arrays. "
            "Use with --prepared-dir to match an existing formal OOF evaluation target."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/real_data/nhanes_p43_final/source/cw_cqr_basis_inference"),
    )
    parser.add_argument("--min-reads", type=int, default=10_000)
    parser.add_argument("--zero-method", default="train_half_q01")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--fold-limit", type=int)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--quantile-levels", type=int, default=20)
    parser.add_argument("--method", choices=("ip", "mm", "admm", "cd"), default="ip")
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--rotation-seed", type=int, action="append", default=[])
    parser.add_argument("--derivative-ridge", type=float, default=1e-10)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument(
        "--r-lib",
        help="Optional R library path; otherwise use CQR_R_LIB/R_LIBS_USER/.libPaths()",
    )
    parser.add_argument("--overwrite-fits", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
