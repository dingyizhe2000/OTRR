"""Build the canonical paper bundle for the final ReLU simulation.

The final experiment combines four PyTorch baselines, CW-CQR, and the
K-reference OT-rank refinement.  This module validates the exact
``noise x method x repetition`` grid and writes one browsable result tree.
Raw fit directories are linked rather than copied.  OT-rank member states are
linked when the optional compact member-state handoff is available locally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from otrank.project_layout import code_root


NOISES = ["A", "B", "C"]
METHODS = ["LS", "CW-Huber", "Radial Huber", "CW-ASM", "CW-CQR", "OTRR"]
METHOD_LABELS = {
    "relu_least_squares": "LS",
    "relu_coordinatewise_huber": "CW-Huber",
    "relu_radial_huber": "Radial Huber",
    "relu_coordinatewise_asm_huber_init": "CW-ASM",
    "relu_coordinatewise_cqr": "CW-CQR",
}
METHOD_SLUGS = {
    "LS": "ls",
    "CW-Huber": "cw_huber",
    "Radial Huber": "radial_huber",
    "CW-ASM": "cw_asm",
    "CW-CQR": "cw_cqr",
    "OTRR": "otrr",
}
BASELINE_ROOT = Path(
    "results/simulation/relu_n1000_p8_d4_final/source/baseline_fits/"
    "relu_prediction_adam_default_float64_n1000_p8_d4_rep50_v1"
)
CQR_ROOT = Path("results/simulation/relu_n1000_p8_d4_final/source/cw_cqr")
OTRR_ROOT = Path("results/simulation/relu_n1000_p8_d4_final/source/otrr")
DEFAULT_MEMBER_ROOT = Path(
    "results/simulation/relu_n1000_p8_d4_final/source/otrr_members/members"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_grid(data: pd.DataFrame) -> None:
    keys = ["noise", "method", "rep"]
    if data.duplicated(keys).any():
        raise ValueError("ReLU metrics contain duplicate noise/method/rep rows")
    expected = {(noise, method): 50 for noise in NOISES for method in METHODS}
    observed = data.groupby(["noise", "method"])["rep"].nunique().to_dict()
    if observed != expected:
        raise ValueError("ReLU metrics are not a complete 3 x 6 x 50 grid")
    numeric = ["prediction_rmse_per_dim", "prediction_mae", "runtime_sec"]
    if not np.isfinite(data[numeric].to_numpy(float)).all():
        raise ValueError("ReLU metrics contain NaN or Inf")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _portable_path(path: Path, project: Path) -> str:
    try:
        return str(path.resolve().relative_to(project))
    except ValueError:
        return str(path.resolve())


def load_metrics(project: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str, int], Path], list[Path]]:
    """Load and validate all six methods and the K=10 member inventory."""

    baseline_root = project / BASELINE_ROOT
    cqr_root = project / CQR_ROOT
    otrr_root = project / OTRR_ROOT
    raw_dirs: dict[tuple[str, str, int], Path] = {}
    sources: list[Path] = []
    rows: list[dict] = []

    metrics_jsonl = baseline_root / "metrics.jsonl"
    sources.append(metrics_jsonl)
    for payload in map(json.loads, metrics_jsonl.read_text().splitlines()):
        method = METHOD_LABELS.get(str(payload["method"]))
        if method is None:
            continue
        noise = str(payload["noise"])
        rep = int(payload["rep_index"])
        fit_dir = next(
            baseline_root.glob(
                f"relu_n1000_p8_d4_{noise}/{payload['method_folder']}/{rep:03d}"
            ),
            None,
        )
        if fit_dir is None or not fit_dir.is_dir():
            raise FileNotFoundError(f"missing baseline fit directory for {(noise, method, rep)}")
        raw_dirs[(noise, method, rep)] = fit_dir
        rows.append(
            {
                "noise": noise,
                "method": method,
                "rep": rep,
                "seed": int(payload["seed"]),
                "prediction_rmse_per_dim": float(payload["prediction_rmse_per_dim"]),
                "prediction_mae": float(payload["prediction_mae"]),
                "runtime_sec": float(payload["runtime_sec"]),
                "source": _portable_path(fit_dir / "metrics.json", project),
            }
        )

    cqr_metric_paths = sorted(cqr_root.glob("relu_n1000_p8_d4_noise*/relu_coordinatewise_cqr/rep*/metrics.json"))
    if len(cqr_metric_paths) != 150:
        raise ValueError(f"expected 150 CW-CQR metrics, found {len(cqr_metric_paths)}")
    for path in cqr_metric_paths:
        payload = _load_json(path)
        noise = str(payload["noise"])
        rep = int(payload["rep_index"])
        raw_dirs[(noise, "CW-CQR", rep)] = path.parent
        rows.append(
            {
                "noise": noise,
                "method": "CW-CQR",
                "rep": rep,
                "seed": int(payload["seed"]),
                "prediction_rmse_per_dim": float(payload["prediction_rmse_per_dim"]),
                "prediction_mae": float(payload["prediction_mae"]),
                "runtime_sec": float(payload["runtime_sec"]),
                "source": _portable_path(path, project),
            }
        )
    sources.extend(cqr_metric_paths)

    otrr_metrics_path = otrr_root / "metrics_by_rep_k.csv"
    otrr_all = pd.read_csv(otrr_metrics_path)
    otrr = otrr_all[otrr_all["K"] == 10].copy()
    if len(otrr) != 150:
        raise ValueError(f"expected 150 final K=10 OTRR rows, found {len(otrr)}")
    for record in otrr.to_dict(orient="records"):
        rows.append(
            {
                "noise": str(record["noise"]),
                "method": "OTRR",
                "rep": int(record["rep"]),
                "seed": int(record["dgp_seed"]),
                "prediction_rmse_per_dim": float(record["otrr_rmse"]),
                "prediction_mae": float(record["otrr_mae"]),
                "runtime_sec": float(record["cumulative_runtime_sec"]),
                "source": _portable_path(otrr_metrics_path, project),
            }
        )

    inventory_path = otrr_root / "fit_inventory.csv"
    inventory = pd.read_csv(inventory_path)
    required_inventory = {
        "noise", "rep", "reference_index", "reference_seed", "m_ref", "state_sha256",
        "initializer_sha256", "reference_sha256", "data_combined_sha256", "status", "finite", "step",
    }
    if not required_inventory.issubset(inventory.columns):
        raise ValueError("OTRR inventory is missing required provenance columns")
    if len(inventory) != 1500:
        raise ValueError(f"expected 1500 OTRR members, found {len(inventory)}")
    inventory["finite"] = inventory["finite"].astype(str).str.lower().eq("true")
    if not (
        inventory["status"].eq("complete").all()
        and inventory["finite"].all()
        and inventory["step"].eq(25).all()
        and inventory["m_ref"].eq(131072).all()
    ):
        raise ValueError("OTRR inventory contains incomplete or nonfinal members")
    expected_counts = {(noise, rep): 10 for noise in NOISES for rep in range(1, 51)}
    observed_counts = inventory.groupby(["noise", "rep"])["reference_index"].nunique().to_dict()
    if observed_counts != expected_counts:
        raise ValueError("OTRR inventory is not a complete 3 x 50 x 10 grid")

    data = pd.DataFrame(rows).sort_values(["noise", "method", "rep"]).reset_index(drop=True)
    _validate_grid(data)
    sources.extend([otrr_metrics_path, inventory_path, otrr_root / "config.json", project / "code/configs/simulation_relu.json"])
    return data, inventory, raw_dirs, sources


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.groupby(["noise", "method"], sort=False)[
        ["prediction_rmse_per_dim", "prediction_mae", "runtime_sec"]
    ].agg(["mean", "std"])
    result.columns = ["_".join(column) for column in result.columns]
    return result.reset_index()


def _relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(os.path.relpath(source, destination.parent))


def _member_path(root: Path, noise: str, rep: int, reference_index: int) -> Path:
    return root / f"noise_{noise}" / f"rep{rep:03d}" / f"K{reference_index:03d}" / "state.pt"


def validate_member_handoff(member_root: Path, inventory: pd.DataFrame) -> tuple[bool, list[Path]]:
    """Validate an optional compact OTRR member-state handoff against its SHA manifest."""

    if not member_root.is_dir():
        return False, []
    paths: list[Path] = []
    for record in inventory.to_dict(orient="records"):
        path = _member_path(member_root, str(record["noise"]), int(record["rep"]), int(record["reference_index"]))
        if not path.is_file():
            raise FileNotFoundError(f"missing OTRR member state: {path}")
        if _sha256(path) != str(record["state_sha256"]):
            raise ValueError(f"OTRR member state SHA-256 mismatch: {path}")
        paths.append(path)
    return True, paths


def build_run_view(
    metrics: pd.DataFrame,
    raw_dirs: dict[tuple[str, str, int], Path],
    member_root: Path,
    member_states_complete: bool,
    output: Path,
) -> pd.DataFrame:
    runs_root = output / "runs"
    if runs_root.exists():
        shutil.rmtree(runs_root)
    inventory_rows: list[dict] = []
    for record in metrics.to_dict(orient="records"):
        noise, method, rep = str(record["noise"]), str(record["method"]), int(record["rep"])
        rep_dir = runs_root / f"noise_{noise}" / METHOD_SLUGS[method] / f"rep{rep:03d}"
        artifacts = rep_dir / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        raw_available = False
        if method == "OTRR" and member_states_complete:
            _relative_symlink(member_root / f"noise_{noise}" / f"rep{rep:03d}", artifacts / "members")
            raw_available = True
        elif method != "OTRR":
            _relative_symlink(raw_dirs[(noise, method, rep)], artifacts / "fit")
            raw_available = True
        payload = {
            "schema_version": 1,
            "experiment": {"model": "relu", "n_train": 1000, "n_test": 5000, "p": 8, "d": 4, "noise": noise, "method": method, "rep": rep},
            "metrics": {key: value for key, value in record.items() if key not in {"noise", "method", "rep"}},
            "artifacts": {
                "raw_fit_available": raw_available,
                "fit_link": "artifacts/members" if method == "OTRR" and raw_available else ("artifacts/fit" if raw_available else None),
                "otrr_reference_count": 10 if method == "OTRR" else None,
            },
        }
        (rep_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
        inventory_rows.append(
            {
                "noise": noise,
                "method": method,
                "method_slug": METHOD_SLUGS[method],
                "rep": rep,
                "result": str((rep_dir / "result.json").relative_to(output)),
                "raw_fit_available": raw_available,
            }
        )
    frame = pd.DataFrame(inventory_rows)
    if len(frame) != 900:
        raise ValueError(f"expected 900 organized ReLU runs, found {len(frame)}")
    frame.to_csv(runs_root / "inventory.csv", index=False)
    return frame


def write_table_rows(summary: pd.DataFrame, path: Path) -> None:
    indexed = summary.set_index(["method", "noise"])
    lines: list[str] = []
    for method in METHODS:
        cells = [method]
        for noise in NOISES:
            row = indexed.loc[(method, noise)]
            cells.append(
                f"{row.prediction_rmse_per_dim_mean:.4f} ({row.prediction_rmse_per_dim_std:.4f})"
            )
        lines.append("        " + " & ".join(cells) + r"\\")
    path.write_text("\n".join(lines) + "\n")


def _manifest(paths: list[Path], project: Path) -> pd.DataFrame:
    rows = []
    for path in sorted({item.resolve() for item in paths}):
        rows.append({"path": _portable_path(path, project), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("results/simulation/relu_n1000_p8_d4_final"))
    parser.add_argument("--otrr-member-root", type=Path, default=DEFAULT_MEMBER_ROOT)
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else project / args.output_dir
    member_root = args.otrr_member_root if args.otrr_member_root.is_absolute() else project / args.otrr_member_root
    for name in ["metrics", "summaries", "tables", "provenance", "config"]:
        (output / name).mkdir(parents=True, exist_ok=True)

    metrics, otrr_inventory, raw_dirs, sources = load_metrics(project)
    member_states_complete, member_paths = validate_member_handoff(member_root, otrr_inventory)
    metrics.to_csv(output / "metrics/prediction_by_rep.csv", index=False)
    otrr_inventory.to_csv(output / "metrics/otrr_reference_inventory.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(output / "summaries/prediction_summary.csv", index=False)
    write_table_rows(summary, output / "tables/relu_prediction_table_rows.tex")
    run_inventory = build_run_view(metrics, raw_dirs, member_root, member_states_complete, output)

    config_source = code_root(project) / "configs/simulation_relu.json"
    shutil.copy2(config_source, output / "config/simulation_relu.json")
    baseline_state_paths = [path / "fit.pt" for path in raw_dirs.values()]
    manifest = _manifest([*sources, *baseline_state_paths, *member_paths], project)
    manifest.to_csv(output / "provenance/source_manifest.csv", index=False)
    environment = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (output / "provenance/analysis_environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    status = "complete" if member_states_complete else "metrics_complete_member_states_pending"
    audit = {
        "status": status,
        "model": "relu_feedforward",
        "n_train": 1000,
        "n_test": 5000,
        "p": 8,
        "d": 4,
        "noises": NOISES,
        "methods": METHODS,
        "repetitions_per_noise_method": 50,
        "prediction_rows": int(len(metrics)),
        "organized_run_rows": int(len(run_inventory)),
        "baseline_fit_states_local": 750,
        "otrr_reference_members": 1500,
        "otrr_member_states_local": int(len(member_paths)),
        "otrr": {"initializer": "CW-CQR", "K": 10, "steps": 25, "m_ref": 131072, "epsilon": 0.1},
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (output / "exact_commands.txt").write_text(
        "PYTHONPATH=code/src python -m otrank.results_analysis.finalize_relu_simulation \\\n"
        "  --project-root . \\\n"
        "  --output-dir results/simulation/relu_n1000_p8_d4_final\n"
    )
    missing_note = (
        "All 1,500 OTRR member states passed SHA-256 validation and are linked below."
        if member_states_complete
        else "The 1,500 OTRR metrics and hashes are complete, but the compact member-state bundle is unavailable. Baseline and CW-CQR fit states are complete."
    )
    (output / "README.md").write_text(
        "# Final ReLU simulation results\n\n"
        "This is the canonical result bundle for `n=1000`, `(p,d)=(8,4)`, Noise A/B/C, and 50 repetitions.\n\n"
        "- `metrics/prediction_by_rep.csv`: the complete 3 x 6 x 50 prediction grid.\n"
        "- `metrics/otrr_reference_inventory.csv`: all 1,500 K=10 member hashes and provenance.\n"
        "- `runs/`: one `noise/method/rep` hierarchy with unified result records and links to raw states.\n"
        "- `summaries/` and `tables/`: regenerated paper-facing values.\n"
        "- `provenance/source_manifest.csv`: content hashes for all locally available source states.\n\n"
        + missing_note + "\n"
    )
    if status == "complete":
        (output / "COMPLETE.marker").write_text("RELU_N1000_P8_D4_FINAL_COMPLETE\n")
    elif (output / "COMPLETE.marker").exists():
        (output / "COMPLETE.marker").unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
