"""Version and rebuild centered linear plug-in inference from retained IFs.

No estimator is refitted and no OT map is re-solved. Small platform differences
in regenerated X are accepted only after common-design reconstruction checks.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from otrank.data.sim import make_linear_simulation
from otrank.inference.centering import recenter_plugin_ensemble_influence
from otrank.inference.covariance import influence_covariance
from otrank.project_layout import code_root
from otrank.results_analysis.finalize_linear_simulation import (
    INFERENCE_METHODS, _validate_grid, summarize, write_table4_rows,
)
from otrank.vectorization import COEFFICIENT_VECTORIZATION


VERSION = "centered_plugin_v1"
ARTIFACT_KIND = "centered_plugin_ensemble_reconstruction"
METRICS = ["avg_interval_length_95", "coverage_95", "avg_interval_length_99", "coverage_99"]


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inference_metrics(phi: np.ndarray, estimate: np.ndarray, truth: np.ndarray) -> dict:
    covariance = influence_covariance(phi)
    se = np.sqrt(np.diag(covariance)/len(phi))
    error = (estimate-truth).reshape(-1, order="F")
    result = {"avg_marginal_standard_error": float(se.mean())}
    for level in (0.8, 0.9, 0.95, 0.99):
        tag = str(round(level*100))
        z = norm.ppf((1+level)/2)
        result[f"avg_interval_length_{tag}"] = float(2*z*se.mean())
        result[f"coverage_{tag}"] = float(np.mean(np.abs(error)<=z*se))
    return result


def directional_metrics(covariance, estimate, truth, n):
    d, p = estimate.shape
    a = np.zeros(p)
    a[:4] = 0.5
    operator = np.kron(a[None], np.eye(d))
    se = np.sqrt(np.diag(operator@covariance@operator.T)/n)
    error = (estimate-truth)@a
    return {
        f"mean_direction_coverage_{level}": float(np.mean(np.abs(error)<=norm.ppf((1+level/100)/2)*se))
        for level in (80, 90, 95, 99)
    }


def _freeze_old_outputs(bundle: Path) -> None:
    destination = bundle/"inference_versions/uncentered_plugin_v1/paper_outputs"
    paths = ["metrics/inference_by_rep.csv", "summaries/inference_summary.csv",
             "tables/table4_rows.tex", "figures/linear_inference_boxplots.pdf",
             "figures/linear_inference_boxplots.png"]
    for name in paths:
        source, target = bundle/name, destination/name
        if source.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def rebuild(project: Path, *, activate: bool, tolerance: float) -> Path:
    torch.set_num_threads(2)
    bundle = project/"results/simulation/linear_n1000_p20_d10_final"
    old_root = bundle/"source/otrr_inference"
    version = bundle/"inference_versions"/VERSION
    version.mkdir(parents=True, exist_ok=True)
    if (version/"COMPLETE.marker").exists():
        raise FileExistsError(f"completed version is immutable: {version}")
    for name in ("metrics", "summaries", "tables", "code_snapshot"):
        (version/name).mkdir(exist_ok=True)
    records, checks, directions = [], [], []
    for noise in "ABC":
        out = version/f"noise_{noise}"
        out.mkdir(exist_ok=True)
        for rep in range(1, 51):
            job_path = bundle/(
                f"source/main/linear_n1000_p20_d10_noise{noise}/ot_rank_asm_init/"
                f"rep{rep:03d}/K001/job_config.json"
            )
            job = json.loads(job_path.read_text())
            data = {k: v for k, v in job["data"].items() if k != "dgp"}
            if (data["n_train"], data["p"], data["d"]) != (1000, 20, 10):
                raise ValueError(f"unexpected linear experiment dimensions in {job_path}")
            if data.get("mu_nonzero") != 4 or data.get("mu_value") != 1:
                raise ValueError("directional checks require the frozen four-coordinate mean")
            simulation = make_linear_simulation(
                **data, noise=job["noise"], seed=job["seed"],
                device="cpu", dtype=torch.float32,
            )
            x = simulation.x_train.double().numpy()
            source = old_root/f"noise_{noise}/rep{rep:03d}_plugin_covariance.npz"
            with np.load(source, allow_pickle=False) as z:
                estimate, truth = z["estimate"].copy(), z["truth"].copy()
                if estimate.shape != (10, 20) or truth.shape != (10, 20):
                    raise ValueError(f"unexpected coefficient shape in {source}")
                result = {}
                diagnostics = {}
                for prefix in ("estimated", "oracle"):
                    raw = z[f"{prefix}_ensemble_influence"].copy()
                    corrected, diag = recenter_plugin_ensemble_influence(
                        x, raw, factorization_tolerance=tolerance,
                    )
                    if not np.allclose(influence_covariance(raw), z[f"{prefix}_covariance"], rtol=1e-10, atol=1e-12):
                        raise ValueError(f"source covariance/IF mismatch: {source}")
                    covariance = influence_covariance(corrected)
                    np.linalg.cholesky(covariance)
                    result[f"{prefix}_ensemble_influence"] = corrected
                    result[f"{prefix}_covariance"] = covariance
                    diagnostics[prefix] = diag
                old_metrics = inference_metrics(z["estimated_ensemble_influence"], estimate, truth)
                old_directions = directional_metrics(z["estimated_covariance"], estimate, truth, len(x))
            target = out/source.name
            np.savez_compressed(
                target, **result, estimate=estimate, truth=truth, x_train=x,
                artifact_kind=np.asarray(ARTIFACT_KIND), artifact_schema_version=np.asarray(1),
                inference_method=np.asarray("plugin_covariance"), inference_version=np.asarray(VERSION),
                vectorization=np.asarray(COEFFICIENT_VECTORIZATION),
                design_centering=np.asarray("sample_mean"), rank_centering=np.asarray("sample_mean"),
                noise=np.asarray(noise), rep=np.asarray(rep),
                n_train=np.asarray(1000), k_references=np.asarray(50),
                source_artifact=np.asarray(str(source.relative_to(project))),
                construction=np.asarray("common_design_kronecker_reconstruction"),
                maps_recomputed=np.asarray(False), coefficients_refitted=np.asarray(False),
                numerically_approximate=np.asarray(True),
                estimated_factorization_relative_error=np.asarray(diagnostics["estimated"]["factorization_relative_error"]),
                oracle_factorization_relative_error=np.asarray(diagnostics["oracle"]["factorization_relative_error"]),
            )
            metrics = inference_metrics(result["estimated_ensemble_influence"], estimate, truth)
            row = {"noise": noise, "rep": rep, "method": "OTRR", **metrics,
                   "source": str(target.relative_to(project))}
            records.append(row)
            original_csv = old_root/f"noise_{noise}/rep{rep:03d}_reference_metrics_plugin_covariance.csv"
            reference_csv = out/original_csv.name
            if not reference_csv.exists():
                shutil.copy2(original_csv, reference_csv)
            metadata = {
                "artifact_kind": ARTIFACT_KIND, "artifact_schema_version": 1,
                "inference_version": VERSION, "inference_method": "plugin_covariance",
                "vectorization": COEFFICIENT_VECTORIZATION,
                "noise": noise, "rep": rep, "n_train": 1000, "K": 50,
                "design_centering": "sample_mean", "rank_centering": "sample_mean",
                "diagnostics": diagnostics, "metrics": metrics,
                "source_artifact": str(source.relative_to(project)),
                "data_seed": int(job["seed"]), "predictor_reconstruction": "recorded_DGP_cpu_float32",
                "platform_bitwise_equality_required": False,
                "original_dual_diagnostics_retained": str(original_csv.relative_to(project)),
                "new_dual_solves": 0, "fits_refitted": 0,
                "finite_temperature_correction_applied": False,
                "approximation": "Tiny source ridge and platform differences controlled by reconstruction tolerance.",
            }
            _json(out/f"rep{rep:03d}_metrics.json", metadata)
            checks.append({"noise": noise, "rep": rep,
                           "estimated_reconstruction_error": diagnostics["estimated"]["factorization_relative_error"],
                           "oracle_reconstruction_error": diagnostics["oracle"]["factorization_relative_error"],
                           **{f"old_{k}": v for k,v in old_metrics.items()},
                           **{f"new_{k}": v for k,v in metrics.items()}})
            directions.extend([
                {"noise": noise, "rep": rep, "version": "uncentered", **old_directions},
                {"noise": noise, "rep": rep, "version": VERSION,
                 **directional_metrics(result["estimated_covariance"], estimate, truth, len(x))},
            ])
        print(f"CENTERED_PLUGIN_COMPLETE noise={noise} reps=50", flush=True)
    original_csv = bundle/"source/paper_inference/inference_plot_data.csv"
    old = pd.read_csv(original_csv)
    corrected = pd.DataFrame(records)
    _validate_grid(corrected, ["OTRR"], "centered OTRR inference")
    required = ["noise", "rep", "method", "avg_marginal_standard_error", *METRICS, "source"]
    combined = pd.concat([old.loc[old.method != "OTRR", required], corrected[required]], ignore_index=True)
    _validate_grid(combined, INFERENCE_METHODS, "centered inference")
    combined.sort_values(["noise", "method", "rep"]).to_csv(version/"metrics/inference_by_rep.csv", index=False)
    corrected.to_csv(version/"metrics/otrr_by_rep.csv", index=False)
    pd.DataFrame(checks).to_csv(version/"metrics/old_vs_centered_by_rep.csv", index=False)
    pd.DataFrame(directions).to_csv(version/"metrics/directional_coverage_by_rep.csv", index=False)
    summary = summarize(combined, METRICS)
    summary.to_csv(version/"summaries/inference_summary.csv", index=False)
    write_table4_rows(summary, version/"tables/table4_rows.tex")
    for file in [Path(__file__), code_root(project)/"src/otrank/inference/centering.py",
                 code_root(project)/"src/otrank/inference/ot.py", code_root(project)/"src/otrank/data/sim.py",
                 code_root(project)/"src/otrank/data/noise.py"]:
        shutil.copy2(file, version/"code_snapshot"/file.name)
    audit = {
        "status": "complete", "artifact_kind": ARTIFACT_KIND, "artifact_schema_version": 1,
        "inference_version": VERSION, "inference_method": "plugin_covariance",
        "vectorization": COEFFICIENT_VECTORIZATION, "design_centering": "sample_mean",
        "rank_centering": "sample_mean", "noise_repetition_counts": {n: 50 for n in "ABC"},
        "new_artifacts": len(records), "new_ot_solves": 0, "new_estimator_fits": 0,
        "estimated_max_reconstruction_error": max(r["estimated_reconstruction_error"] for r in checks),
        "oracle_max_reconstruction_error": max(r["oracle_reconstruction_error"] for r in checks),
        "factorization_tolerance": tolerance, "numerically_approximate": True,
        "finite_temperature_correction_applied": False,
        "source_covariance_root": "source/otrr_inference",
        "covariance_normalization": "V=IF.T@IF/n; estimator covariance=V/n",
    }
    _json(version/"audit.json", audit)
    (version/"README.md").write_text(
        "# Centered linear plug-in inference, version 1\n\n"
        "A/B/C each contain 50 K=50 ensemble artifacts, with estimated and oracle IFs, "
        "covariances, unchanged coefficients/truth, and the reconstructed design. "
        "No fits or OT maps were recomputed. Sample-centered design/rank scores were "
        "recovered using the checked common-design Kronecker structure. The reconstruction "
        "is numerical: tiny source-ridge and platform differences are bounded by the "
        "recorded residual tolerance. Original artifacts and dual diagnostics are retained. "
        "This does not implement a finite-temperature full-Hessian correction.\n\n"
        "The metrics directory contains all per-repetition interval lengths, coverage, "
        "old/new comparisons, and design-mean-direction coverage. Parentheses in paper "
        "tables are repetition standard deviations, not MCSE.\n"
    )
    (version/"exact_commands.txt").write_text(" ".join(sys.argv)+"\n")
    (version/"COMPLETE.marker").write_text("CENTERED_PLUGIN_COMPLETE_150_OF_150\n")
    if activate:
        _freeze_old_outputs(bundle)
        _json(bundle/"config/active_inference.json", {
            "schema_version": 1, "version": VERSION,
            "root": str(version.relative_to(bundle)), "artifact_kind": ARTIFACT_KIND,
            "design_centering": "sample_mean", "rank_centering": "sample_mean",
        })
    print(f"Saved {version}; active={activate}", flush=True)
    return version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--activate", action="store_true", help="Select this completed version for paper outputs.")
    parser.add_argument("--factorization-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    rebuild(args.project_root.resolve(), activate=args.activate, tolerance=args.factorization_tolerance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
