"""Read only the explicitly selected, compact current-paper metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from otrank.project_layout import code_root

PAPER_VERSION = "centered_equivariance_v1"
FILES = {
    "linear_point.csv", "linear_equivariance.csv", "linear_inference.csv",
    "relu_prediction.csv", "nhanes_summary.csv", "nhanes_equivariance.csv",
    "nhanes_pairwise.csv", "nhanes_predictors.csv", "semisynthetic_repetitions.csv",
    "semisynthetic_ratio.csv",
}


def data_root(project: Path | None = None) -> Path:
    source = Path(__file__).resolve().parents[3] if project is None else code_root(project)
    return source / "paper_data"


def load_frame(name: str, project: Path | None = None) -> pd.DataFrame:
    if name not in FILES:
        raise ValueError(f"not an approved paper-metric file: {name}")
    root = data_root(project)
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Current paper_data/manifest.json is required; no archive or legacy fallback is allowed.")
    manifest = json.loads(path.read_text())
    if (manifest.get("paper_version") != PAPER_VERSION
            or manifest.get("linear_inference_version") != "centered_plugin_v1"
            or manifest.get("equivariance_metric") != "entrywise_max_abs"):
        raise ValueError("paper data do not select the current centered/equivariance version")
    records = manifest.get("files", [])
    if len(records) != len(FILES) or {entry["file"] for entry in records} != FILES:
        raise ValueError("paper data inventory is not the explicit current-paper whitelist")
    entry = next(item for item in records if item["file"] == name)
    frame = pd.read_csv(root / name)
    if list(frame.columns) != entry["columns"] or len(frame) != entry["rows"]:
        raise ValueError(f"paper metric schema/count mismatch: {name}")
    if not np.isfinite(frame.select_dtypes(include="number").to_numpy()).all():
        raise ValueError(f"nonfinite paper metrics: {name}")
    if any("angle" in column for column in frame):
        raise ValueError("historical geometric metrics are not current paper outputs")
    return frame


def validate_grid(frame: pd.DataFrame, methods: list[str], repetitions: int = 50) -> None:
    keys = ["noise", "method", "rep"]
    expected = {(noise, method, rep) for noise in "ABC" for method in methods for rep in range(1, repetitions+1)}
    if frame.duplicated(keys).any() or set(frame[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("paper metrics do not contain the complete frozen repetition grid")


def linear_point(project: Path | None = None) -> pd.DataFrame:
    point = load_frame("linear_point.csv", project)
    equivariance = load_frame("linear_equivariance.csv", project)
    methods = ["LS", "Radial Huber", "CW-Huber", "CW-ASM", "CW-CQR", "MCQR", "OTRR"]
    for frame in (point, equivariance):
        validate_grid(frame, methods)
    return point.merge(equivariance, on=["noise", "method", "rep"], validate="one_to_one", sort=False)


def nhanes_equivariance(project: Path | None = None) -> pd.DataFrame:
    frame = load_frame("nhanes_equivariance.csv", project)
    methods = ["LS", "Radial Huber", "CW-Huber", "CW-ASM", "CW-CQR", "OTRR"]
    keys = ["method", "fold", "rotation_seed"]
    expected = {(method, fold, seed) for method in methods for fold in range(1, 6) for seed in (20260731, 20260732)}
    if frame.duplicated(keys).any() or set(frame[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("NHANES equivariance requires five folds and both rotations for all methods")
    return frame.groupby("method", sort=False).e_max.agg(e_max_mean="mean", e_max_sd="std").reset_index()


def equivariance_cell(mean: float, sd: float, threshold: float = 1e-6) -> str:
    """Format table entries only; never censor the data used by plots."""
    if mean < threshold:
        return rf"$<10^{{{int(np.log10(threshold))}}}$"
    digits = 6 if mean < 0.001 else 5
    return rf"${mean:.{digits}f}\,({sd:.{digits}f})$"
