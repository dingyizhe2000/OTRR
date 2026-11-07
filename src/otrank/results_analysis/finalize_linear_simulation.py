"""Create the self-contained paper bundle for the final linear simulation.

The builder reads the source records stored inside the canonical result bundle,
extracts only the final ``n=1000, p=20, d=10`` and ``K=50`` estimators, and
writes compact per-repetition CSVs plus paper-ready tables and figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from otrank.results_analysis.plot_linear_inference_95_99 import (
    build_figure as build_inference_figure,
)
from otrank.project_layout import code_root
from otrank.results_analysis.paper_data import equivariance_cell, linear_point as current_linear_point
from otrank.results_analysis.plot_linear_accuracy_equivariance import build_figure as build_equivariance_figure


NOISES = ["A", "B", "C"]
INFERENCE_METHODS = ["LS", "Radial Huber", "CW-Huber", "CW-ASM", "CW-CQR", "OTRR"]
POINT_METHODS = [
    "LS",
    "Radial Huber",
    "CW-Huber",
    "CW-ASM",
    "CW-CQR",
    "MCQR",
    "OTRR",
]
BASELINE_LABELS = {
    "zero_intercept_ols": "LS",
    "radial_huber_tau1p345_sqrtd": "Radial Huber",
    "coordinatewise_huber_tau1p345": "CW-Huber",
    "coordinatewise_asm_zero_intercept": "CW-ASM",
    "mcqr_ot_lens": "MCQR",
}
METHOD_SLUGS = {
    "LS": "ls",
    "Radial Huber": "radial_huber",
    "CW-Huber": "cw_huber",
    "CW-ASM": "cw_asm",
    "CW-CQR": "cw_cqr",
    "MCQR": "mcqr",
    "OTRR": "otrr",
}
MAIN_METHOD_FOLDERS = {
    "LS": "zero_intercept_ols",
    "Radial Huber": "radial_huber_tau1p345_sqrtd",
    "CW-Huber": "coordinatewise_huber_tau1p345",
    "CW-ASM": "coordinatewise_asm_zero_intercept",
    "MCQR": "mcqr_ot_lens",
    "OTRR": "ot_rank_asm_init",
}


def format_table_value(value: float) -> str:
    """Format paper values to three decimals using half-up rounding."""

    decimal_value = Decimal(f"{float(value):.12f}")
    return format(decimal_value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP), ".3f")
COLORS = {
    "LS": "#88CCEE",
    "Radial Huber": "#117733",
    "CW-Huber": "#44AA99",
    "CW-ASM": "#CCBB44",
    "CW-CQR": "#AA4499",
    "MCQR": "#999933",
    "OTRR": "#332288",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_grid(data: pd.DataFrame, methods: list[str], name: str) -> None:
    keys = ["noise", "method", "rep"]
    if data.duplicated(keys).any():
        raise ValueError(f"{name} contains duplicate noise/method/rep rows")
    expected = {(noise, method): 50 for noise in NOISES for method in methods}
    observed = data.groupby(["noise", "method"])["rep"].nunique().to_dict()
    if observed != expected:
        raise ValueError(f"{name} is not a complete 3 x {len(methods)} x 50 grid")


def load_point_metrics(project: Path) -> tuple[pd.DataFrame, list[Path]]:
    baseline_path = (
        project
        / "results/simulation/linear_n1000_p20_d10_final/source/baselines/"
        "simulation_existing_baselines.csv"
    )
    cqr_path = (
        project
        / "results/simulation/linear_n1000_p20_d10_final/source/"
        "cw_cqr/repetition_metrics.csv"
    )
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[
        (baseline["n_train"] == 1000)
        & (baseline["p"] == 20)
        & (baseline["d"] == 10)
        & baseline["noise"].isin(NOISES)
        & baseline["method"].isin(BASELINE_LABELS)
        & baseline["rep_index"].between(1, 50)
    ].copy()
    baseline["method"] = baseline["method"].map(BASELINE_LABELS)
    baseline["rep"] = baseline["rep_index"].astype(int)
    baseline["max_angle_deg"] = np.degrees(
        pd.to_numeric(baseline["rotation_coef_principal_angle_max"], errors="raise")
    )
    baseline = baseline.sort_values("line_no").drop_duplicates(
        ["noise", "method", "rep"], keep="last"
    )
    baseline["source"] = str(baseline_path.relative_to(project))
    baseline = baseline[
        [
            "noise",
            "rep",
            "seed",
            "method",
            "coef_error",
            "centered_prediction_mse",
            "prediction_mse",
            "prediction_mae",
            "max_angle_deg",
            "rotation_coef_principal_angle_fro_sin",
            "source",
        ]
    ]

    cqr = pd.read_csv(cqr_path)
    cqr = cqr[
        cqr["noise"].isin(NOISES)
        & cqr["rep"].between(1, 50)
        & (cqr["n_train"] == 1000)
        & (cqr["p"] == 20)
        & (cqr["d"] == 10)
    ].copy()
    cqr["method"] = "CW-CQR"
    cqr["max_angle_deg"] = cqr["rotation_coef_principal_angle_max_deg"]
    cqr["source"] = str(cqr_path.relative_to(project))
    cqr = cqr[
        [
            "noise",
            "rep",
            "seed",
            "method",
            "coef_error",
            "centered_prediction_mse",
            "prediction_mse",
            "prediction_mae",
            "max_angle_deg",
            "rotation_coef_principal_angle_fro_sin",
            "source",
        ]
    ]

    otrr_rows: list[dict] = []
    otrr_sources: list[Path] = []
    main_root = project / "results/simulation/linear_n1000_p20_d10_final/source/main"
    for noise in NOISES:
        for rep in range(1, 51):
            path = (
                main_root
                / f"linear_n1000_p20_d10_noise{noise}/ot_rank_asm_init/"
                f"rep{rep:03d}/ensemble_metrics.json"
            )
            payload = json.loads(path.read_text())["metrics"]["K050"]["iter500"]
            if int(payload["ensemble_k"]) != 50 or int(payload["iteration"]) != 500:
                raise ValueError(f"unexpected OTRR ensemble in {path}")
            otrr_rows.append(
                {
                    "noise": noise,
                    "rep": rep,
                    "seed": int(payload["seed"]),
                    "method": "OTRR",
                    "coef_error": float(payload["coef_error"]),
                    "centered_prediction_mse": float(payload["centered_prediction_mse"]),
                    "prediction_mse": float(payload["prediction_mse"]),
                    "prediction_mae": float(payload["prediction_mae"]),
                    "max_angle_deg": math.degrees(
                        float(payload["rotation_coef_principal_angle_max"])
                    ),
                    "rotation_coef_principal_angle_fro_sin": float(
                        payload["rotation_coef_principal_angle_fro_sin"]
                    ),
                    "source": str(path.relative_to(project)),
                }
            )
            otrr_sources.append(path)

    data = pd.concat([baseline, cqr, pd.DataFrame(otrr_rows)], ignore_index=True)
    _validate_grid(data, POINT_METHODS, "point metrics")
    numeric = [
        "coef_error",
        "centered_prediction_mse",
        "prediction_mse",
        "prediction_mae",
        "max_angle_deg",
        "rotation_coef_principal_angle_fro_sin",
    ]
    if not np.isfinite(data[numeric].to_numpy(float)).all():
        raise ValueError("point metrics contain NaN or Inf")
    cqr_states = sorted(cqr_path.parent.glob("noise_*/rep*/fit_and_inference.npz"))
    if len(cqr_states) != 150:
        raise ValueError(f"expected 150 CW-CQR states, found {len(cqr_states)}")
    current = current_linear_point(project)
    data = data.drop(columns=[column for column in data if "angle" in column or "fro_sin" in column])
    data = data.merge(current[["noise", "method", "rep", "coef_error", "e_max"]],
                      on=["noise", "method", "rep"], validate="one_to_one", suffixes=("", "_current"))
    if not np.allclose(data.coef_error, data.coef_error_current, rtol=1e-12, atol=1e-14):
        raise ValueError("retained point estimates do not match the current paper version")
    data = data.drop(columns="coef_error_current")
    return data.sort_values(["noise", "method", "rep"]), [
        baseline_path,
        cqr_path,
        *cqr_states,
        *otrr_sources,
    ]


def selected_inference_root(project: Path) -> Path:
    """Select a completed version explicitly, retaining historical inputs."""
    bundle = project / "results/simulation/linear_n1000_p20_d10_final"
    selector = bundle / "config/active_inference.json"
    if not selector.exists():
        raise FileNotFoundError("Current centered inference selector is missing; run the documented migration, not a legacy fallback.")
    config = json.loads(selector.read_text())
    root = (bundle / config["root"]).resolve()
    if not root.is_relative_to(bundle.resolve()) or not (root/"COMPLETE.marker").is_file():
        raise ValueError("active inference must be a completed version inside the linear bundle")
    audit = json.loads((root/"audit.json").read_text())
    if (audit.get("inference_version") != config.get("version")
            or audit.get("design_centering") != "sample_mean"
            or audit.get("rank_centering") != "sample_mean"
            or audit.get("new_artifacts") != 150):
        raise ValueError("active centered inference metadata does not match the selected version")
    return root


def load_inference_metrics(project: Path) -> tuple[pd.DataFrame, list[Path]]:
    covariance_root = selected_inference_root(project)
    path = covariance_root / "metrics/inference_by_rep.csv"
    if not path.exists():
        if covariance_root.name != "otrr_inference":
            raise FileNotFoundError(f"selected inference has no combined metrics: {path}")
        path = project / (
            "results/simulation/linear_n1000_p20_d10_final/source/"
            "paper_inference/inference_plot_data.csv"
        )
    data = pd.read_csv(path)
    _validate_grid(data, INFERENCE_METHODS, "inference metrics")
    required = [
        "avg_marginal_standard_error",
        "avg_interval_length_95",
        "coverage_95",
        "avg_interval_length_99",
        "coverage_99",
    ]
    if not np.isfinite(data[required].to_numpy(float)).all():
        raise ValueError("inference metrics contain NaN or Inf")
    if covariance_root.name != "otrr_inference":
        for row in data.loc[data.method == "OTRR"].itertuples(index=False):
            artifact = covariance_root / f"noise_{row.noise}/rep{row.rep:03d}_plugin_covariance.npz"
            with np.load(artifact, allow_pickle=False) as z:
                expected = {"artifact_kind": "centered_plugin_ensemble_reconstruction",
                            "inference_method": "plugin_covariance", "design_centering": "sample_mean",
                            "rank_centering": "sample_mean", "vectorization": "column-major vec(B)",
                            "noise": row.noise, "rep": row.rep}
                if any(key not in z or z[key].item() != value for key, value in expected.items()):
                    raise ValueError(f"selected centered artifact metadata mismatch: {artifact}")
                phi, covariance = z["estimated_ensemble_influence"], z["estimated_covariance"]
                if phi.shape != (1000, 200) or covariance.shape != (200, 200):
                    raise ValueError(f"selected centered artifact shape mismatch: {artifact}")
                if not np.isfinite(phi).all() or not np.isfinite(covariance).all():
                    raise ValueError(f"selected centered artifact is not finite: {artifact}")
                centered_phi = phi-phi.mean(0)
                if not np.allclose(covariance, centered_phi.T@centered_phi/len(phi), rtol=1e-10, atol=1e-12):
                    raise ValueError(f"selected covariance does not match its saved IF: {artifact}")
                se = np.sqrt(np.diag(covariance)/len(phi))
                error = (z["estimate"]-z["truth"]).reshape(-1, order="F")
                for level in (95, 99):
                    quantile = norm.ppf((1+level/100)/2)
                    length = float(2*quantile*se.mean())
                    coverage = float(np.mean(np.abs(error)<=quantile*se))
                    if (not np.isclose(length, getattr(row, f"avg_interval_length_{level}"), rtol=1e-10, atol=1e-12)
                            or not np.isclose(coverage, getattr(row, f"coverage_{level}"), atol=1e-12, rtol=0)):
                        raise ValueError(f"selected metrics do not match covariance and estimate: {artifact}")
    retained = ["noise", "rep", "method", *required]
    if "source" in data:
        retained.append("source")
    data = data.loc[:, retained].copy()
    if "source" in data:
        def portable_source(value: object) -> str:
            source = Path(str(value))
            try:
                return str(source.resolve().relative_to(project))
            except ValueError:
                return str(source)

        data["source"] = data["source"].map(portable_source)

    covariance_artifacts = sorted(covariance_root.glob("noise_*/*"))
    expected_suffix_counts = {
        "_plugin_covariance.npz": 150,
        "_metrics.json": 150,
        "_reference_metrics_plugin_covariance.csv": 150,
    }
    for suffix, expected in expected_suffix_counts.items():
        observed = sum(path.name.endswith(suffix) for path in covariance_artifacts)
        if observed != expected:
            raise ValueError(f"expected {expected} OTRR artifacts ending in {suffix}, found {observed}")
    root_files = [path for path in covariance_root.iterdir() if path.is_file()]
    paper_files = [item for item in path.parent.iterdir() if item.is_file()]
    return data.sort_values(["noise", "method", "rep"]), [
        *paper_files,
        *root_files,
        *covariance_artifacts,
    ]


def summarize(data: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    result = data.groupby(["noise", "method"], sort=False)[metrics].agg(["mean", "std"])
    result.columns = ["_".join(column) for column in result.columns]
    return result.reset_index()


def _relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(os.path.relpath(source, destination.parent))


def _json_record(row: pd.Series) -> dict:
    record: dict = {}
    for key, value in row.items():
        if pd.isna(value):
            record[key] = None
        elif isinstance(value, (np.integer, int)):
            record[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            record[key] = float(value)
        else:
            record[key] = str(value)
    return record


def build_run_view(
    point: pd.DataFrame,
    inference: pd.DataFrame,
    project: Path,
    output: Path,
) -> pd.DataFrame:
    """Create one noise/method/rep view without duplicating heavy artifacts."""

    runs_root = output / "runs"
    if runs_root.exists():
        shutil.rmtree(runs_root)
    runs_root.mkdir(parents=True)
    inference_index = inference.set_index(["noise", "method", "rep"])
    inventory: list[dict] = []

    main_root = project / "results/simulation/linear_n1000_p20_d10_final/source/main"
    cqr_root = (
        project
        / "results/simulation/linear_n1000_p20_d10_final/source/cw_cqr"
    )
    covariance_root = selected_inference_root(project)

    for _, point_row in point.sort_values(["noise", "method", "rep"]).iterrows():
        noise = str(point_row["noise"])
        method = str(point_row["method"])
        rep = int(point_row["rep"])
        rep_dir = runs_root / f"noise_{noise}" / METHOD_SLUGS[method] / f"rep{rep:03d}"
        artifacts_dir = rep_dir / "artifacts"
        artifacts_dir.mkdir(parents=True)

        key = (noise, method, rep)
        inference_record = (
            _json_record(inference_index.loc[key])
            if key in inference_index.index
            else None
        )
        raw_fit: Path | None = None
        if method == "CW-CQR":
            candidate = cqr_root / f"noise_{noise}" / f"rep{rep:03d}"
            raw_fit = candidate if candidate.is_dir() else None
        elif method in MAIN_METHOD_FOLDERS:
            candidate = (
                main_root
                / f"linear_n1000_p20_d10_noise{noise}"
                / MAIN_METHOD_FOLDERS[method]
                / f"rep{rep:03d}"
            )
            raw_fit = candidate if candidate.is_dir() else None
        if raw_fit is not None:
            _relative_symlink(raw_fit, artifacts_dir / "fit")

        covariance_available = False
        if method == "OTRR":
            covariance_files = {
                "plugin_covariance.npz": covariance_root
                / f"noise_{noise}/rep{rep:03d}_plugin_covariance.npz",
                "plugin_metrics.json": covariance_root
                / f"noise_{noise}/rep{rep:03d}_metrics.json",
                "reference_diagnostics.csv": covariance_root
                / f"noise_{noise}/rep{rep:03d}_reference_metrics_plugin_covariance.csv",
            }
            if not all(path.is_file() for path in covariance_files.values()):
                raise ValueError(f"incomplete OTRR inference artifacts for {key}")
            for name, source in covariance_files.items():
                _relative_symlink(source, artifacts_dir / name)
            covariance_available = True

        payload = {
            "schema_version": 1,
            "experiment": {
                "model": "linear",
                "n_train": 1000,
                "n_test": 5000,
                "p": 20,
                "d": 10,
                "noise": noise,
                "method": method,
                "rep": rep,
            },
            "point_estimation": _json_record(point_row),
            "inference": inference_record,
            "artifacts": {
                "raw_fit_available": raw_fit is not None,
                "fit_link": "artifacts/fit" if raw_fit is not None else None,
                "plugin_covariance_available": covariance_available,
            },
        }
        (rep_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
        inventory.append(
            {
                "noise": noise,
                "method": method,
                "method_slug": METHOD_SLUGS[method],
                "rep": rep,
                "result": str((rep_dir / "result.json").relative_to(output)),
                "raw_fit_available": raw_fit is not None,
                "inference_available": inference_record is not None,
                "plugin_covariance_available": covariance_available,
            }
        )

    inventory_frame = pd.DataFrame(inventory)
    if len(inventory_frame) != 1050:
        raise ValueError(f"expected 1050 organized runs, found {len(inventory_frame)}")
    inventory_frame.to_csv(runs_root / "inventory.csv", index=False)
    (runs_root / "README.md").write_text(
        "# Organized linear simulation runs\n\n"
        "Use only this directory to browse per-repetition results. The hierarchy is "
        "`noise_<A|B|C>/<method>/repXXX/`. Every repetition has one `result.json` "
        "containing final point-estimation and, when available, inference metrics.\n\n"
        "The `artifacts/` entries are relative symbolic links to audited source states, "
        "so this view does not duplicate large checkpoints. A missing `fit` link means "
        "that the original fit state was not available locally; the final per-repetition "
        "metrics remain recorded in `result.json`.\n"
    )
    return inventory_frame


def write_table3_rows(summary: pd.DataFrame, path: Path) -> None:
    indexed = summary.set_index(["noise", "method"])
    minima = {
        (noise, metric): min(
            float(indexed.loc[(noise, method), f"{metric}_mean"])
            for method in POINT_METHODS
        )
        for noise in NOISES
        for metric in ["coef_error"]
    }
    lines: list[str] = []
    for method in POINT_METHODS:
        cells = [method]
        for noise in NOISES:
            row = indexed.loc[(noise, method)]
            coef = f"{row.coef_error_mean:.3f} ({row.coef_error_std:.3f})"
            equivariance = equivariance_cell(row.e_max_mean, row.e_max_std)
            if np.isclose(row.coef_error_mean, minima[(noise, "coef_error")], atol=5e-13):
                coef = f"\\textbf{{{coef}}}"
            cells.extend([coef, equivariance])
        lines.append("        " + " & ".join(cells) + r"\\")
    path.write_text("\n".join([*lines, "\\hline"]) + "\n")


def write_table4_rows(summary: pd.DataFrame, path: Path) -> None:
    indexed = summary.set_index(["noise", "method"])
    specs = [
        ("95\\% length", "avg_interval_length_95", True),
        ("95\\% cov.", "coverage_95", False),
        ("99\\% length", "avg_interval_length_99", True),
        ("99\\% cov.", "coverage_99", False),
    ]
    lines: list[str] = []
    for noise in NOISES:
        for index, (label, metric, bold_minimum) in enumerate(specs):
            values = [float(indexed.loc[(noise, method), f"{metric}_mean"]) for method in INFERENCE_METHODS]
            minimum = min(values)
            cells: list[str] = []
            for method, value in zip(INFERENCE_METHODS, values):
                sd = float(indexed.loc[(noise, method), f"{metric}_std"])
                cell = f"{format_table_value(value)} ({format_table_value(sd)})"
                if bold_minimum and np.isclose(value, minimum, atol=5e-13):
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
            prefix = f"\\multirow{{4}}{{*}}{{{noise}}} & " if index == 0 else "  & "
            lines.append(prefix + label + " & " + " & ".join(cells) + r"\\")
        lines.append("\\hline")
    path.write_text("\n".join(lines) + "\n")


def build_accuracy_figure(data: pd.DataFrame) -> plt.Figure:
    return build_equivariance_figure(data)


def _manifest(paths: list[Path], project: Path) -> pd.DataFrame:
    unique = sorted({path.resolve() for path in paths})
    rows = []
    for path in unique:
        rows.append(
            {
                "path": str(path.relative_to(project)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/simulation/linear_n1000_p20_d10_final"),
    )
    parser.add_argument(
        "--inference-only", action="store_true",
        help="Rebuild inference outputs only; leave fits, point metrics and Figure S2.1 unchanged.",
    )
    parser.add_argument(
        "--manuscript-dir",
        type=Path,
        default=None,
        help="Optional manuscript root receiving generated tables and figures.",
    )
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else project / args.output_dir
    manuscript = None
    if args.manuscript_dir is not None:
        manuscript = (
            args.manuscript_dir
            if args.manuscript_dir.is_absolute()
            else project / args.manuscript_dir
        )
        (manuscript / "tables").mkdir(parents=True, exist_ok=True)
        (manuscript / "figures").mkdir(parents=True, exist_ok=True)
    metrics_dir = output / "metrics"
    summaries_dir = output / "summaries"
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    provenance_dir = output / "provenance"
    config_dir = output / "config"
    for directory in [metrics_dir, summaries_dir, tables_dir, figures_dir, provenance_dir, config_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    if args.inference_only:
        inference, _ = load_inference_metrics(project)
        inference.to_csv(metrics_dir / "inference_by_rep.csv", index=False)
        summary = summarize(inference, ["avg_interval_length_95", "coverage_95", "avg_interval_length_99", "coverage_99"])
        summary.to_csv(summaries_dir / "inference_summary.csv", index=False)
        write_table4_rows(summary, tables_dir / "table4_rows.tex")
        figure = build_inference_figure(inference)
        figure.savefig(figures_dir / "linear_inference_boxplots.pdf", bbox_inches="tight")
        figure.savefig(figures_dir / "linear_inference_boxplots.png", dpi=350, bbox_inches="tight")
        plt.close(figure)
        root = selected_inference_root(project)
        for row in inference.loc[inference.method == "OTRR"].itertuples(index=False):
            rep_dir = output / f"runs/noise_{row.noise}/otrr/rep{row.rep:03d}"
            result_path = rep_dir / "result.json"
            if result_path.exists():
                payload = json.loads(result_path.read_text())
                payload["inference"] = _json_record(pd.Series(row._asdict()))
                result_path.write_text(json.dumps(payload, indent=2)+"\n")
                for name, source in {
                    "plugin_covariance.npz": root / f"noise_{row.noise}/rep{row.rep:03d}_plugin_covariance.npz",
                    "plugin_metrics.json": root / f"noise_{row.noise}/rep{row.rep:03d}_metrics.json",
                    "reference_diagnostics.csv": root / f"noise_{row.noise}/rep{row.rep:03d}_reference_metrics_plugin_covariance.csv",
                }.items():
                    _relative_symlink(source, rep_dir / "artifacts" / name)
        update = {"inference_version": root.name, "rows": len(inference),
                  "source": str(root.relative_to(project)), "coefficient_fits_changed": False,
                  "point_metrics_changed": False, "figure_s2_1_changed": False}
        (provenance_dir / "centered_inference_update.json").write_text(json.dumps(update, indent=2)+"\n")
        audit_path = output / "audit.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            audit.update({"inference_version": root.name, "design_centering": "sample_mean",
                          "rank_centering": "sample_mean"})
            audit_path.write_text(json.dumps(audit, indent=2)+"\n")
        if manuscript is not None:
            shutil.copy2(tables_dir / "table4_rows.tex", manuscript / "tables/linear_inference_table_rows.tex")
            shutil.copy2(figures_dir / "linear_inference_boxplots.pdf", manuscript / "figures/linear_inference_boxplots.pdf")
        print(f"Rebuilt centered inference only from {root}")
        return 0

    point, point_sources = load_point_metrics(project)
    inference, inference_sources = load_inference_metrics(project)
    point.to_csv(metrics_dir / "point_estimation_by_rep.csv", index=False)
    inference.to_csv(metrics_dir / "inference_by_rep.csv", index=False)
    run_inventory = build_run_view(point, inference, project, output)
    point_summary = summarize(point, ["coef_error", "e_max"])
    inference_summary = summarize(
        inference,
        ["avg_interval_length_95", "coverage_95", "avg_interval_length_99", "coverage_99"],
    )
    point_summary.to_csv(summaries_dir / "point_estimation_summary.csv", index=False)
    inference_summary.to_csv(summaries_dir / "inference_summary.csv", index=False)

    write_table3_rows(point_summary, tables_dir / "table3_rows.tex")
    write_table4_rows(inference_summary, tables_dir / "table4_rows.tex")
    if manuscript is not None:
        shutil.copy2(
            tables_dir / "table3_rows.tex",
            manuscript / "tables/linear_accuracy_table_rows.tex",
        )
        shutil.copy2(
            tables_dir / "table4_rows.tex",
            manuscript / "tables/linear_inference_table_rows.tex",
        )

    accuracy_figure = build_accuracy_figure(point)
    accuracy_targets = [figures_dir / "linear_accuracy_invariance_boxplots.pdf"]
    if manuscript is not None:
        accuracy_targets.append(manuscript / "figures/linear_accuracy_invariance_boxplots.pdf")
    for target in accuracy_targets:
        accuracy_figure.savefig(target, bbox_inches="tight")
    for target in [figures_dir / "linear_accuracy_invariance_boxplots.png"]:
        accuracy_figure.savefig(target, dpi=350, bbox_inches="tight")
    plt.close(accuracy_figure)

    inference_figure = build_inference_figure(inference)
    inference_targets = [figures_dir / "linear_inference_boxplots.pdf"]
    if manuscript is not None:
        inference_targets.append(manuscript / "figures/linear_inference_boxplots.pdf")
    for target in inference_targets:
        inference_figure.savefig(target, bbox_inches="tight")
    for target in [figures_dir / "linear_inference_boxplots.png"]:
        inference_figure.savefig(target, dpi=350, bbox_inches="tight")
    plt.close(inference_figure)

    source_root = code_root(project)
    config_source = source_root / "configs/simulation_linear.json"
    config = json.loads(config_source.read_text())
    if config.get("sample_sizes") != [1000]:
        raise ValueError("the frozen linear paper config must contain only n=1000")
    shutil.copy2(config_source, config_dir / "simulation_linear.json")
    analysis_sources = [
        source_root / "src/otrank/results_analysis/finalize_linear_simulation.py",
        source_root / "src/otrank/results_analysis/plot_linear_inference_95_99.py",
    ]
    manifest = _manifest(
        [config_source, *analysis_sources, *point_sources, *inference_sources], project
    )
    manifest.to_csv(provenance_dir / "source_manifest.csv", index=False)

    environment = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": plt.matplotlib.__version__,
    }
    (provenance_dir / "analysis_environment.json").write_text(
        json.dumps(environment, indent=2) + "\n"
    )

    audit = {
        "status": "complete",
        "model": "linear",
        "n_train": 1000,
        "n_test": 5000,
        "p": 20,
        "d": 10,
        "noises": NOISES,
        "point_methods": POINT_METHODS,
        "inference_methods": INFERENCE_METHODS,
        "repetitions_per_noise_method": 50,
        "point_rows": int(len(point)),
        "inference_rows": int(len(inference)),
        "organized_run_rows": int(len(run_inventory)),
        "otrr": {"K": 50, "steps": 500, "m_ref": 524288, "epsilon": 0.1},
        "inference_method": "plugin_covariance",
        "inference_version": selected_inference_root(project).name,
        "vectorization": "column-major vec(B)",
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (output / "exact_commands.txt").write_text(
        "MPLCONFIGDIR=.mplconfig PYTHONPATH=code/src \\\n"
        "python -m otrank.results_analysis.finalize_linear_simulation \\\n"
        "  --project-root . \\\n"
        "  --output-dir results/simulation/linear_n1000_p20_d10_final\n"
    )
    (output / "README.md").write_text(
        "# Final linear simulation results\n\n"
        "This is the canonical paper bundle for the linear simulation with "
        "`n=1000`, `p=20`, `d=10`, Noise A/B/C, and 50 Monte Carlo repetitions.\n\n"
        "- `metrics/point_estimation_by_rep.csv`: 3 x 7 x 50 final point-estimation rows.\n"
        "- `metrics/inference_by_rep.csv`: 3 x 6 x 50 final inference rows.\n"
        "- `runs/`: one `noise/method/rep` hierarchy with unified `result.json` records.\n"
        "- `summaries/`: paper means and repetition-level standard deviations.\n"
        "- `tables/` and `figures/`: paper-ready outputs regenerated from the CSVs.\n"
        "- `provenance/source_manifest.csv`: source paths, sizes, and SHA-256 hashes.\n"
        "- `provenance/analysis_environment.json`: table/figure environment versions.\n"
        "- `config/simulation_linear.json`: frozen paper configuration.\n\n"
        "The `source/` directory contains the retained formal input records used by this builder; "
        "no dated experiment directory outside this bundle is required.\n"
    )
    (output / "COMPLETE.marker").write_text("LINEAR_N1000_P20_D10_FINAL_COMPLETE\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
