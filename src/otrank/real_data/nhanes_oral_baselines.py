"""Penalized and unpenalized baselines for the NHANES oral microbiome."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from otrank.estimators.penalized_cv import (
    PENALIZED_METHODS,
    fit_penalized_linear_method,
    select_l2_penalty,
)
from otrank.estimators.penalized_linear import LinearASMConfig
from otrank.real_data.preprocessing import (
    common_response_scale,
    fit_preprocessor,
    select_full_rank_columns,
    transform_response,
)


DEFAULT_DATA_ROOT = Path("data/nhanes_oral_microbiome/raw")
DEFAULT_OUT = Path("results/nhanes_oral_core6_baselines")
DEFAULT_PENALTIES = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0)

CORE_GENERA = {
    "Actinomyces": "RSV_genus35",
    "Rothia": "RSV_genus135",
    "Prevotella 7": "RSV_genus263",
    "Gemella": "RSV_genus475",
    "Streptococcus": "RSV_genus530",
    "Veillonella": "RSV_genus740",
}

CYCLES = (
    ("2009_2010", "F"),
    ("2011_2012", "G"),
)

TABLES = (
    "DEMO",
    "BMX",
    "SMQ",
    "ALQ",
    "DIQ",
    "BPQ",
    "GHB",
    "DR1TOT",
    "OHQ",
    "OHXDEN",
    "OHXPER",
    "RXQ_RX",
)


def _read_xpt(path: Path) -> pd.DataFrame:
    frame = pd.read_sas(path, format="xport")
    if "SEQN" not in frame:
        raise ValueError(f"NHANES table has no SEQN column: {path}")
    frame = frame.copy()
    frame["SEQN"] = frame["SEQN"].astype(np.int64)
    return frame


def _load_cycle_tables(root: Path) -> dict[str, pd.DataFrame]:
    combined: dict[str, list[pd.DataFrame]] = {name: [] for name in TABLES}
    for cycle, suffix in CYCLES:
        cycle_root = root / f"nhanes_{cycle}"
        for table in TABLES:
            frame = _read_xpt(cycle_root / f"{table}_{suffix}.XPT")
            frame["cycle"] = cycle
            combined[table].append(frame)
    return {name: pd.concat(parts, ignore_index=True) for name, parts in combined.items()}


def _finite_codes(values: pd.Series, allowed: set[int]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(numeric.isin(allowed))


def _bounded(values: pd.Series, low: float, high: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(numeric.between(low, high))


def _yes_no(values: pd.Series) -> pd.Series:
    coded = _finite_codes(values, {1, 2})
    return coded.map({1.0: 1.0, 2.0: 0.0})


def _dummy_columns(
    values: pd.Series,
    *,
    prefix: str,
    levels: tuple[int, ...],
    reference: int,
) -> pd.DataFrame:
    numeric = _finite_codes(values, set(levels))
    out = pd.DataFrame(index=values.index)
    for level in levels:
        if level != reference:
            out[f"{prefix}_{level}"] = (numeric == level).astype(float)
    out[f"{prefix}_missing"] = numeric.isna().astype(float)
    return out


def _cycle_aggregate(table: pd.DataFrame) -> pd.DataFrame:
    if table["SEQN"].duplicated().any():
        raise ValueError("Expected one row per SEQN")
    return table.set_index("SEQN").drop(columns="cycle", errors="ignore")


def _periodontal_summary(table: pd.DataFrame) -> pd.DataFrame:
    indexed = table.set_index("SEQN")
    out = pd.DataFrame(index=indexed.index)
    for label, pattern in (
        ("pocket_depth", re.compile(r"^OHX\d{2}PC[A-Z]$")),
        ("attachment_loss", re.compile(r"^OHX\d{2}LA[A-Z]$")),
    ):
        columns = [column for column in indexed if pattern.match(str(column))]
        values = indexed[columns].apply(pd.to_numeric, errors="coerce")
        values = values.where(values < 90.0)
        out[f"{label}_mean"] = values.mean(axis=1)
        out[f"{label}_max"] = values.max(axis=1)
        threshold = 4.0
        valid = values.notna().sum(axis=1)
        out[f"{label}_fraction_ge4"] = (values >= threshold).sum(axis=1).div(
            valid.where(valid > 0)
        )
    return out


def _dentition_summary(table: pd.DataFrame) -> pd.DataFrame:
    indexed = table.set_index("SEQN")
    tooth_columns = [
        column for column in indexed if re.match(r"^OHX\d{2}TC$", str(column))
    ]
    values = indexed[tooth_columns].apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame(index=indexed.index)
    out["natural_tooth_count"] = values.isin([1.0, 2.0]).sum(axis=1).astype(float)
    out["implant_count"] = (values == 3.0).sum(axis=1).astype(float)
    out["root_fragment_count"] = (values == 5.0).sum(axis=1).astype(float)
    out.loc[values.notna().sum(axis=1) == 0, :] = np.nan
    return out


def _prescription_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for seqn, frame in table.groupby("SEQN", sort=False):
        use = _finite_codes(frame["RXDUSE"], {1, 2})
        if bool((use == 1).any()):
            count = pd.to_numeric(frame["RXDCOUNT"], errors="coerce").max()
        elif bool((use == 2).any()):
            count = 0.0
        else:
            count = np.nan
        rows.append({"SEQN": int(seqn), "prescription_count": float(count)})
    return pd.DataFrame(rows).set_index("SEQN")


def make_predictors(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    demo = tables["DEMO"].set_index("SEQN")
    out = pd.DataFrame(index=demo.index)

    age = _bounded(demo["RIDAGEYR"], 18.0, 69.0)
    out["age"] = age
    out["age_squared"] = age**2
    out["age_cubed"] = age.pow(3)
    out["male"] = _finite_codes(demo["RIAGENDR"], {1, 2}).map({1.0: 1.0, 2.0: 0.0})
    out["income_poverty_ratio"] = _bounded(demo["INDFMPIR"], 0.0, 5.0)
    out["cycle_2011_2012"] = (demo["cycle"] == "2011_2012").astype(float)
    out = out.join(
        _dummy_columns(
            demo["RIDRETH1"], prefix="race", levels=(1, 2, 3, 4, 5), reference=3
        )
    )
    out = out.join(
        _dummy_columns(
            demo["DMDEDUC2"],
            prefix="education",
            levels=(1, 2, 3, 4, 5),
            reference=5,
        )
    )
    out = out.join(
        _dummy_columns(
            demo["DMDMARTL"],
            prefix="marital",
            levels=(1, 2, 3, 4, 5, 6),
            reference=1,
        )
    )

    bmx = _cycle_aggregate(tables["BMX"])
    out["bmi"] = _bounded(bmx["BMXBMI"], 10.0, 80.0)
    out["waist_cm"] = _bounded(bmx["BMXWAIST"], 40.0, 200.0)

    smoking = _cycle_aggregate(tables["SMQ"])
    ever = _yes_no(smoking["SMQ020"])
    current_code = _finite_codes(smoking["SMQ040"], {1, 2, 3})
    out["smoking_ever"] = ever
    out["smoking_current"] = pd.Series(
        np.where(
            ever == 0.0,
            0.0,
            np.where(
                current_code.isin([1.0, 2.0]),
                1.0,
                np.where(current_code == 3.0, 0.0, np.nan),
            ),
        ),
        index=smoking.index,
    )

    alcohol = _cycle_aggregate(tables["ALQ"])
    out["alcohol_12plus_year"] = _yes_no(alcohol["ALQ101"])
    out["alcohol_drinks_per_day"] = _bounded(alcohol["ALQ130"], 0.0, 25.0)

    diabetes = _cycle_aggregate(tables["DIQ"])
    diabetes_code = _finite_codes(diabetes["DIQ010"], {1, 2, 3})
    out["diabetes"] = diabetes_code.map({1.0: 1.0, 2.0: 0.0, 3.0: 0.0})
    out["borderline_diabetes"] = diabetes_code.map({1.0: 0.0, 2.0: 0.0, 3.0: 1.0})

    blood_pressure = _cycle_aggregate(tables["BPQ"])
    out["hypertension"] = _yes_no(blood_pressure["BPQ020"])
    hba1c = _cycle_aggregate(tables["GHB"])
    out["hba1c"] = _bounded(hba1c["LBXGH"], 2.0, 20.0)

    diet = _cycle_aggregate(tables["DR1TOT"])
    diet_variables = {
        "diet_kcal": ("DR1TKCAL", 0.0, 15000.0),
        "diet_sugar_g": ("DR1TSUGR", 0.0, 2000.0),
        "diet_fiber_g": ("DR1TFIBE", 0.0, 200.0),
        "diet_fat_g": ("DR1TTFAT", 0.0, 1000.0),
        "diet_alcohol_g": ("DR1TALCO", 0.0, 500.0),
    }
    for name, (column, low, high) in diet_variables.items():
        out[f"log1p_{name}"] = np.log1p(_bounded(diet[column], low, high))

    oral = _cycle_aggregate(tables["OHQ"])
    out["self_reported_gum_disease"] = _yes_no(oral["OHQ835"])
    out["oral_health_rating_worse"] = _bounded(oral["OHQ845"], 1.0, 5.0)
    out["gum_disease_treatment"] = _yes_no(oral["OHQ850"])
    out["loose_teeth"] = _yes_no(oral["OHQ855"])
    out["bone_loss"] = _yes_no(oral["OHQ860"])
    out["abnormal_tooth_recent"] = _yes_no(oral["OHQ865"])
    out["floss_days_week"] = _bounded(oral["OHQ870"], 0.0, 7.0)

    out = out.join(_dentition_summary(tables["OHXDEN"]))
    out = out.join(_periodontal_summary(tables["OHXPER"]))
    out = out.join(_prescription_summary(tables["RXQ_RX"]))
    return out


def load_analysis_data(
    root: Path,
    *,
    min_reads: int = 10_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, dict[str, object]]:
    tables = _load_cycle_tables(root)
    counts = pd.read_csv(
        root / "oral_microbiome" / "dada2rsv-genus-count.txt", sep="\t"
    ).dropna(subset=["SEQN"])
    relative = pd.read_csv(
        root / "oral_microbiome" / "dada2rsv-genus-relative.txt", sep="\t"
    ).dropna(subset=["SEQN"])
    for frame in (counts, relative):
        frame["SEQN"] = frame["SEQN"].astype(np.int64)

    count_values = counts.drop(columns="SEQN").to_numpy(dtype=np.float64)
    reads = pd.Series(count_values.sum(axis=1), index=counts["SEQN"], name="reads")
    demo = tables["DEMO"].set_index("SEQN")
    cohort = demo.index[
        demo["RIDAGEYR"].between(18.0, 69.0)
        & (demo["RIDSTATR"] == 2.0)
        & (reads.reindex(demo.index) >= int(min_reads))
    ]
    relative = relative.set_index("SEQN").loc[cohort]
    predictors = make_predictors(tables).loc[cohort]

    missing_columns = sorted(set(CORE_GENERA.values()) - set(relative.columns))
    if missing_columns:
        raise ValueError(f"Missing requested RSV genus columns: {missing_columns}")
    response = relative[list(CORE_GENERA.values())].copy()
    response.columns = list(CORE_GENERA)
    total = relative.sum(axis=1)
    response["Other"] = (total - response.sum(axis=1)).clip(lower=0.0)
    if bool((response.sum(axis=1) <= 0.0).any()):
        raise ValueError("At least one core-six response has zero total mass")

    strata = (
        demo.loc[cohort, "cycle"].astype(str)
        + "_race"
        + demo.loc[cohort, "RIDRETH1"].fillna(0).astype(int).astype(str)
    )
    cohort_frame = demo.loc[
        cohort,
        ["cycle", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"],
    ].copy()
    cohort_frame.insert(0, "reads", reads.loc[cohort])
    coverage = response.drop(columns="Other").sum(axis=1)
    audit = {
        "n": int(len(cohort)),
        "min_reads": int(min_reads),
        "response_components": list(response.columns),
        "response_dim": int(response.shape[1] - 1),
        "predictor_columns_raw": list(predictors.columns),
        "predictor_dim_raw": int(predictors.shape[1]),
        "core_coverage_mean": float(coverage.mean()),
        "core_coverage_median": float(coverage.median()),
        "cycle_counts": cohort_frame["cycle"].value_counts().sort_index().to_dict(),
        "taxonomy_resolution": "public RSV abundance is aggregated at genus level",
        "core_genus_ids": CORE_GENERA,
    }
    return predictors, response, cohort_frame, strata, audit


def _method_label(method: str, penalized: bool) -> str:
    base = {
        "least_squares": "LS",
        "coordinatewise_huber": "CW-Huber",
        "radial_huber": "Radial-Huber",
        "coordinatewise_asm": "CW-ASM",
    }[method]
    return f"{base} + L2-CV" if penalized else base


def run(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(args.numeric_threads)))
    predictors, response, cohort, strata, audit = load_analysis_data(
        args.data_root, min_reads=int(args.min_reads)
    )
    methods = tuple(args.method or PENALIZED_METHODS)
    unknown = set(methods) - set(PENALIZED_METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    outer = StratifiedKFold(
        n_splits=int(args.outer_folds), shuffle=True, random_state=int(args.seed)
    )
    outer_splits = list(outer.split(np.arange(len(predictors)), strata.to_numpy()))
    asm_config = LinearASMConfig(
        alt_iter=int(args.asm_alt_iter),
        max_iter=int(args.asm_max_iter),
        score_grid_size=int(args.asm_score_grid_size),
        kernel_points=int(args.asm_kernel_points),
    )

    metrics_rows: list[dict[str, object]] = []
    cv_rows: list[dict[str, object]] = []
    split_audit: list[dict[str, object]] = []
    continuous = set(predictors.columns)
    for split_zero, (train_idx, test_idx) in enumerate(outer_splits):
        split = split_zero + 1
        x_train_frame, x_test_frame = fit_preprocessor(
            predictors.iloc[train_idx], predictors.iloc[test_idx], continuous
        )
        x_train_frame, x_test_frame = select_full_rank_columns(
            x_train_frame, x_test_frame
        )
        y_train_np, y_test_np, zero_audit = transform_response(
            response.iloc[train_idx], response.iloc[test_idx], args.zero_method
        )
        x_train = torch.as_tensor(x_train_frame.to_numpy(), dtype=torch.float64)
        x_test = torch.as_tensor(x_test_frame.to_numpy(), dtype=torch.float64)
        y_train_raw = torch.as_tensor(y_train_np, dtype=torch.float64)
        y_test = torch.as_tensor(y_test_np, dtype=torch.float64)
        response_scale = common_response_scale(x_train, y_train_raw)
        y_train = y_train_raw / response_scale

        inner = StratifiedKFold(
            n_splits=int(args.inner_folds),
            shuffle=True,
            random_state=int(args.seed) + 1000 * split,
        )
        inner_folds = list(
            inner.split(np.arange(len(train_idx)), strata.iloc[train_idx].to_numpy())
        )
        mean_prediction = y_train_raw.mean(dim=0, keepdim=True).expand(
            y_test.shape[0], -1
        )
        baseline_sse = float(torch.sum((y_test - mean_prediction).square()).item())

        metrics_rows.append(
            {
                "split": split,
                "method": "Mean-only",
                "method_key": "mean_only",
                "penalized": False,
                "selected_penalty": 0.0,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "predictor_dim": x_train.shape[1],
                "response_dim": y_train.shape[1],
                "rmse_per_ilr_dim": math.sqrt(
                    baseline_sse / (y_test.shape[0] * y_test.shape[1])
                ),
                "test_r2": 0.0,
                "sse": baseline_sse,
                "baseline_sse": baseline_sse,
                "coef_fro_norm": 0.0,
                "cv_runtime_sec": 0.0,
                "fit_runtime_sec": 0.0,
            }
        )

        for method in methods:
            cv_started = time.perf_counter()
            cv_result = select_l2_penalty(
                method,
                x_train,
                y_train,
                inner_folds,
                args.penalty,
                asm_config=asm_config,
                workers=int(args.cv_workers),
            )
            cv_runtime = time.perf_counter() - cv_started
            for penalty, scores in cv_result.fold_validation_mse.items():
                for inner_fold, score in enumerate(scores, start=1):
                    cv_rows.append(
                        {
                            "outer_split": split,
                            "method": method,
                            "penalty": penalty,
                            "inner_fold": inner_fold,
                            "validation_mse_scaled": score,
                            "selected": penalty == cv_result.selected_penalty,
                        }
                    )

            for penalized, penalty in (
                (False, 0.0),
                (True, cv_result.selected_penalty),
            ):
                fit_started = time.perf_counter()
                fit = fit_penalized_linear_method(
                    method,
                    x_train,
                    y_train,
                    penalty,
                    asm_config=asm_config,
                )
                fit_runtime = time.perf_counter() - fit_started
                prediction = fit.predict(x_test) * response_scale
                residual = y_test - prediction
                sse = float(torch.sum(residual.square()).item())
                metrics_rows.append(
                    {
                        "split": split,
                        "method": _method_label(method, penalized),
                        "method_key": method,
                        "penalized": penalized,
                        "selected_penalty": penalty,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "predictor_dim": x_train.shape[1],
                        "response_dim": y_train.shape[1],
                        "rmse_per_ilr_dim": float(
                            torch.sqrt(torch.mean(residual.square())).item()
                        ),
                        "test_r2": 1.0 - sse / baseline_sse,
                        "sse": sse,
                        "baseline_sse": baseline_sse,
                        "coef_fro_norm": float(torch.linalg.norm(fit.coef).item()),
                        "cv_runtime_sec": cv_runtime if penalized else 0.0,
                        "fit_runtime_sec": fit_runtime,
                    }
                )
            print(
                f"done split={split}/{args.outer_folds} method={method} "
                f"lambda={cv_result.selected_penalty:g} cv_sec={cv_runtime:.2f}",
                flush=True,
            )

        zero_audit.update(
            {
                "split": split,
                "predictor_columns": list(x_train_frame.columns),
                "response_scale": response_scale,
            }
        )
        split_audit.append(zero_audit)

    metrics = pd.DataFrame(metrics_rows)
    summary = (
        metrics.groupby("method", sort=False)
        .agg(
            folds=("split", "count"),
            rmse_mean=("rmse_per_ilr_dim", "mean"),
            rmse_sd=("rmse_per_ilr_dim", "std"),
            test_r2_mean=("test_r2", "mean"),
            test_r2_median=("test_r2", "median"),
            sse_sum=("sse", "sum"),
            baseline_sse_sum=("baseline_sse", "sum"),
            penalty_median=("selected_penalty", "median"),
            coef_fro_norm_mean=("coef_fro_norm", "mean"),
            cv_runtime_sec_mean=("cv_runtime_sec", "mean"),
            fit_runtime_sec_mean=("fit_runtime_sec", "mean"),
        )
        .reset_index()
    )
    summary["pooled_test_r2"] = 1.0 - summary["sse_sum"] / summary["baseline_sse_sum"]
    metrics.to_csv(args.out / "fold_metrics.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(args.out / "cv_scores.csv", index=False)
    cohort.to_csv(args.out / "cohort_audit.csv")
    response.describe(percentiles=[0.05, 0.5, 0.95]).T.to_csv(
        args.out / "response_summary.csv"
    )
    pd.DataFrame(
        {
            "missing_fraction": predictors.isna().mean(),
            "mean": predictors.mean(),
            "sd": predictors.std(ddof=0),
        }
    ).to_csv(args.out / "predictor_audit.csv")
    audit.update(
        {
            "zero_method": args.zero_method,
            "outer_folds": int(args.outer_folds),
            "inner_folds": int(args.inner_folds),
            "penalties": list(map(float, args.penalty)),
            "methods": list(methods),
            "splits": split_audit,
        }
    )
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-reads", type=int, default=10_000)
    parser.add_argument("--zero-method", default="train_half_q01")
    parser.add_argument("--method", action="append", choices=PENALIZED_METHODS)
    parser.add_argument("--penalty", type=float, action="append", default=[])
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--cv-workers", type=int, default=4)
    parser.add_argument("--numeric-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--asm-alt-iter", type=int, default=1)
    parser.add_argument("--asm-max-iter", type=int, default=65)
    parser.add_argument("--asm-score-grid-size", type=int, default=3000)
    parser.add_argument("--asm-kernel-points", type=int, default=8192)
    args = parser.parse_args()
    if not args.penalty:
        args.penalty = list(DEFAULT_PENALTIES)
    return args


if __name__ == "__main__":
    run(parse_args())
