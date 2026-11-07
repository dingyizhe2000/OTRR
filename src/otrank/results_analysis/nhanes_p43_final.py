"""Finalize the canonical NHANES p=43, d=7 paper results.

This module is deliberately analysis-only: it validates fixed fitted states,
reconstructs summaries, and creates a navigable result index.  It never fits or
updates an estimator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from otrank.results_analysis.paper_data import equivariance_cell, nhanes_equivariance, load_frame


METHODS = ("LS", "Radial Huber", "CW-Huber", "CW-ASM", "CW-CQR", "OTRR")
METHOD_IDS = {
    "LS": "ols",
    "Radial Huber": "radial_huber",
    "CW-Huber": "cw_huber",
    "CW-ASM": "r_asm",
    "CW-CQR": "cw_cqr",
}
ROTATIONS = (20260731, 20260732)
REFERENCE_SEEDS = tuple(range(20260721, 20260731))
COMPONENTS = (
    "Actinomyces",
    "Rothia",
    "Prevotella 7",
    "Gemella",
    "Streptococcus",
    "Veillonella",
    "Haemophilus",
    "Other",
)
Z_975 = 1.959963984540054

OTRR_MAIN_DIR = "formal_cw_cqr_K10"
OTRR_ROTATED_INFERENCE_DIR = "formal_cw_cqr_rotated_inference"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def _helmert_rows(parts: int = 8) -> np.ndarray:
    basis = np.zeros((parts - 1, parts), dtype=np.float64)
    for row in range(parts - 1):
        width = row + 2
        basis[row, : width - 1] = 1.0 / math.sqrt((width - 1) * width)
        basis[row, width - 1] = -(width - 1) / math.sqrt((width - 1) * width)
    return basis


def _prepared(root: Path, fold: int) -> dict[str, np.ndarray]:
    path = root / "source/baselines/train_half_q01/prepared" / f"fold_{fold:02d}.npz"
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def _effect_multipliers(prepared: dict[str, np.ndarray]) -> np.ndarray:
    x_train = np.asarray(prepared["X_train"], dtype=np.float64)
    names = np.asarray(prepared["predictor_names"]).astype(str)
    result = np.ones(x_train.shape[1], dtype=np.float64)
    for index, name in enumerate(names):
        levels = np.unique(x_train[:, index])
        if len(levels) == 2:
            result[index] = levels[1] - levels[0]
        elif name == "oral_health_rating_worse":
            result[index] = float(np.median(np.diff(levels)))
    return result


def _otrr_roots(root: Path) -> tuple[Path, Path]:
    otrr = root / "source/otrr"
    return otrr / OTRR_MAIN_DIR, otrr / OTRR_ROTATED_INFERENCE_DIR


def validate_sources(root: Path) -> dict[str, Any]:
    main, rotated = _otrr_roots(root)
    required = (
        root / "source/baselines/oof_prediction_summary.csv",
        root / "source/baseline_basis_stability/strict_stability_summary.csv",
        root / "source/cw_cqr_basis_inference/COMPLETE.marker",
        main / "cw_cqr_only_summary/COMPLETE.marker",
        rotated / "COMPLETE.marker",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    fit_rows = []
    for fold in range(1, 6):
        prepared = _prepared(root, fold)
        expected = (6589 if fold <= 2 else 6590, 43)
        if prepared["X_train"].shape != expected or prepared["Y_train"].shape != (
            expected[0],
            7,
        ):
            raise ValueError(f"unexpected prepared shape for fold {fold}")
        if tuple(prepared["composition_components"].astype(str)) != COMPONENTS:
            raise ValueError(f"unexpected composition order for fold {fold}")
        for representation in ("helmert", *[f"rotation_{s}" for s in ROTATIONS]):
            for reference, seed in enumerate(REFERENCE_SEEDS, 1):
                if representation == "helmert":
                    state = main / (
                        f"original_fits/fold_{fold:02d}/cw_cqr/"
                        f"reference_{reference:02d}_seed{seed}/iteration_500/state.pt"
                    )
                else:
                    state = main / (
                        f"rotation_fits/fold_{fold:02d}/cw_cqr/{representation}/"
                        f"reference_{reference:02d}_seed{seed}/iteration_500/state.pt"
                    )
                if not state.is_file():
                    raise FileNotFoundError(state)
                fit_rows.append((fold, representation, reference, seed, state))
        inference = main / f"inference/fold_{fold:02d}/cw_cqr/K10/plugin_covariance.npz"
        with np.load(inference, allow_pickle=False) as source:
            if source["estimate"].shape != (7, 43) or source["covariance"].shape != (
                301,
                301,
            ):
                raise ValueError(f"unexpected inference shape for fold {fold}")
            if str(source["inference_method"].item()) != "plugin_covariance":
                raise ValueError(f"unexpected inference method for fold {fold}")
            if str(source["parameter_order"].item()) != "column-major vec(B)":
                raise ValueError(f"unexpected parameter order for fold {fold}")
            if not np.isfinite(source["covariance"]).all():
                raise ValueError(f"non-finite covariance for fold {fold}")
    return {
        "folds": 5,
        "representations": 3,
        "otrr_reference_members": len(fit_rows),
        "expected_otrr_reference_members": 150,
        "source_validation": "passed",
    }


def _load_otrr_predictions(root: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    main, _ = _otrr_roots(root)
    fold_rows: list[dict[str, float | int | str]] = []
    pooled: dict[str, list[np.ndarray]] = {"residual": [], "baseline": []}
    for fold in range(1, 6):
        prepared = _prepared(root, fold)
        predictions = []
        train_means = []
        for reference, seed in enumerate(REFERENCE_SEEDS, 1):
            state_path = main / (
                f"original_fits/fold_{fold:02d}/cw_cqr/"
                f"reference_{reference:02d}_seed{seed}/iteration_500/state.pt"
            )
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            coefficient = np.asarray(state["coef"], dtype=np.float64)
            x_mean = np.asarray(state["x_mean"], dtype=np.float64)
            y_mean = np.asarray(state["y_mean"], dtype=np.float64)
            predictions.append((prepared["X_test"] - x_mean) @ coefficient.T + y_mean)
            train_means.append(y_mean)
        prediction = np.mean(predictions, axis=0)
        train_mean = np.mean(train_means, axis=0)
        residual = np.asarray(prepared["Y_test"], dtype=np.float64) - prediction
        baseline = np.asarray(prepared["Y_test"], dtype=np.float64) - train_mean
        row_error = np.linalg.norm(residual, axis=1)
        fold_rows.append(
            {
                "fold": fold,
                "method": "OTRR",
                "n_test": len(residual),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "mae": float(np.mean(np.abs(residual))),
                "r2": float(1.0 - np.sum(residual**2) / np.sum(baseline**2)),
                "median_row_error": float(np.median(row_error)),
                "q90_row_error": float(np.quantile(row_error, 0.90)),
                "q95_row_error": float(np.quantile(row_error, 0.95)),
                "q99_row_error": float(np.quantile(row_error, 0.99)),
                "max_row_error": float(np.max(row_error)),
            }
        )
        pooled["residual"].append(residual)
        pooled["baseline"].append(baseline)
    return pd.DataFrame(fold_rows), {key: np.concatenate(value) for key, value in pooled.items()}


def _prediction_bootstrap(
    residual: np.ndarray,
    baseline: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    n, d = residual.shape
    squared = np.sum(residual**2, axis=1)
    absolute = np.mean(np.abs(residual), axis=1)
    baseline_squared = np.sum(baseline**2, axis=1)
    draws = np.empty((repetitions, 3), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for index in range(repetitions):
        selected = rng.integers(0, n, size=n)
        sse = squared[selected].sum()
        draws[index] = (
            math.sqrt(sse / (n * d)),
            absolute[selected].mean(),
            1.0 - sse / baseline_squared[selected].sum(),
        )
    return {
        "rmse_bootstrap_sd": float(draws[:, 0].std(ddof=1)),
        "mae_bootstrap_sd": float(draws[:, 1].std(ddof=1)),
        "r2_bootstrap_sd": float(draws[:, 2].std(ddof=1)),
    }


def build_prediction_summary(
    root: Path, *, bootstrap_repetitions: int, bootstrap_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_csv(root / "source/baselines/oof_prediction_summary.csv")
    name_map = {
        "ols": "LS",
        "radial_huber": "Radial Huber",
        "cw_huber": "CW-Huber",
        "r_asm": "CW-ASM",
        "cw_cqr": "CW-CQR",
    }
    baseline["method"] = baseline["method"].map(name_map)
    otrr_folds, pooled = _load_otrr_predictions(root)
    residual, denominator = pooled["residual"], pooled["baseline"]
    row_error = np.linalg.norm(residual, axis=1)
    otrr = {
        "zero_rule": "train_half_q01",
        "method": "OTRR",
        "n": len(residual),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - np.sum(residual**2) / np.sum(denominator**2)),
        "median_sample_error": float(np.median(row_error)),
        "q90_sample_error": float(np.quantile(row_error, 0.90)),
        "q95_sample_error": float(np.quantile(row_error, 0.95)),
        "q99_sample_error": float(np.quantile(row_error, 0.99)),
        "max_sample_error": float(np.max(row_error)),
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_seed": bootstrap_seed,
        **_prediction_bootstrap(
            residual,
            denominator,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed,
        ),
    }
    summary = pd.concat([baseline, pd.DataFrame([otrr])], ignore_index=True)
    summary["method"] = pd.Categorical(summary["method"], METHODS, ordered=True)
    return summary.sort_values("method"), otrr_folds


def _otrr_contrast_outputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    main, _ = _otrr_roots(root)
    basis = _helmert_rows()
    contrast_rows: list[dict[str, Any]] = []
    fold_ci = []
    for fold in range(1, 6):
        prepared = _prepared(root, fold)
        names = prepared["predictor_names"].astype(str)
        multipliers = _effect_multipliers(prepared)
        n_train = len(prepared["X_train"])
        path = main / f"inference/fold_{fold:02d}/cw_cqr/K10/plugin_covariance.npz"
        with np.load(path, allow_pickle=False) as source:
            coefficient = np.asarray(source["estimate"], dtype=np.float64)
            covariance = np.asarray(source["covariance"], dtype=np.float64)
        lengths = []
        p_values = []
        temporary = []
        for predictor in range(43):
            block = covariance[
                predictor * 7 : (predictor + 1) * 7,
                predictor * 7 : (predictor + 1) * 7,
            ]
            for genus in range(7):
                difference = np.zeros(8)
                difference[genus], difference[-1] = 1.0, -1.0
                contrast = basis @ difference
                multiplier = multipliers[predictor]
                estimate = float(multiplier * contrast @ coefficient[:, predictor])
                variance = max(float(contrast @ block @ contrast), 0.0)
                se = float(multiplier * math.sqrt(variance / n_train))
                p_value = math.erfc(abs(estimate / se) / math.sqrt(2.0)) if se else 0.0
                length = 2.0 * Z_975 * se
                lengths.append(length)
                p_values.append(p_value)
                temporary.append((predictor, genus, multiplier, estimate, se, length, p_value))
        q_values = _bh_adjust(np.asarray(p_values))
        for values, q_value in zip(temporary, q_values, strict=True):
            predictor, genus, multiplier, estimate, se, length, p_value = values
            contrast_rows.append(
                {
                    "fold": fold,
                    "representation": "helmert",
                    "method": "OTRR",
                    "predictor_index": predictor,
                    "predictor": names[predictor],
                    "genus_index": genus,
                    "genus": COMPONENTS[genus],
                    "effect_multiplier": multiplier,
                    "estimate": estimate,
                    "standard_error": se,
                    "ci95_length": length,
                    "p_value": p_value,
                    "bh_q_value": float(q_value),
                }
            )
        fold_ci.append({"fold": fold, "average_95_ci_length": float(np.mean(lengths))})
    return pd.DataFrame(contrast_rows), pd.DataFrame(fold_ci)


def _fold_bootstrap_sd(values: np.ndarray, *, seed: int, repetitions: int) -> float:
    rng = np.random.default_rng(seed)
    draws = [values[rng.integers(0, len(values), size=len(values))].mean() for _ in range(repetitions)]
    return float(np.std(draws, ddof=1))


def _strict_sets(data: pd.DataFrame, *, method: str | None = None) -> dict[str, set[tuple[int, int]]]:
    if method is not None:
        data = data[data["method"].eq(method)]
    result: dict[str, set[tuple[int, int]]] = {}
    for representation, group in data.groupby("representation"):
        selected: set[tuple[int, int]] = set()
        for key, rows in group.groupby(["predictor_index", "genus_index"]):
            estimates = rows["estimate"].to_numpy()
            if (
                len(rows) == 5
                and (rows["bh_q_value"].to_numpy() < 0.05).all()
                and estimates[0] != 0
                and (np.sign(estimates) == np.sign(estimates[0])).all()
            ):
                selected.add((int(key[0]), int(key[1])))
        result[str(representation)] = selected
    return result


def build_stability_outputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_csv(root / "source/baseline_basis_stability/genus_vs_other_contrasts.csv").rename(
        columns={"q_value": "bh_q_value"}
    )
    cqr = pd.read_csv(root / "source/cw_cqr_basis_inference/genus_vs_other_contrasts.csv").rename(
        columns={"q_value": "bh_q_value"}
    )
    _, rotated = _otrr_roots(root)
    otrr = pd.read_csv(rotated / "genus_vs_other_contrasts.csv")
    sources = {
        "LS": _strict_sets(baseline, method="ols"),
        "Radial Huber": _strict_sets(baseline, method="radial_huber"),
        "CW-Huber": _strict_sets(baseline, method="cw_huber"),
        "CW-ASM": _strict_sets(baseline, method="r_asm"),
        "CW-CQR": _strict_sets(cqr),
        "OTRR": _strict_sets(otrr),
    }
    representations = ("helmert", "rotation_20260731", "rotation_20260732")
    summary_rows, pair_rows = [], []
    for method, sets in sources.items():
        values = [sets[name] for name in representations]
        intersection, union = set.intersection(*values), set.union(*values)
        summary_rows.append(
            {
                "method": method,
                **{f"strict_{name}": len(sets[name]) for name in representations},
                "intersection": len(intersection),
                "union": len(union),
                "intersection_over_union": len(intersection) / len(union),
                "stable_rule": "same sign and foldwise BH q<0.05 in all five folds",
            }
        )
        for left_index in range(3):
            for right_index in range(left_index + 1, 3):
                left, right = representations[left_index], representations[right_index]
                common, combined = sets[left] & sets[right], sets[left] | sets[right]
                pair_rows.append(
                    {
                        "method": method,
                        "representation_1": left,
                        "representation_2": right,
                        "intersection": len(common),
                        "union": len(combined),
                        "intersection_over_union": len(common) / len(combined),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)


def _angle_summary(
    root: Path, *, bootstrap_seed: int, bootstrap_repetitions: int
) -> pd.DataFrame:
    baseline = pd.read_csv(root / "source/baselines/table5_baseline_update/table5_baseline_rows.csv")
    baseline_map = {
        "ols": "LS",
        "radial_huber": "Radial Huber",
        "cw_huber": "CW-Huber",
        "r_asm": "CW-ASM",
        "cw_cqr": "CW-CQR",
    }
    rows = [
        {
            "method": baseline_map[row.method],
            "max_angle_deg": row.max_angle_deg,
            "max_angle_bootstrap_sd_deg": row.max_angle_bootstrap_sd_deg,
        }
        for row in baseline.itertuples(index=False)
    ]
    main, _ = _otrr_roots(root)
    invariance = pd.read_csv(main / "cw_cqr_only_summary/invariance_metrics.csv")
    values = invariance[
        invariance["method"].eq("otrr_cw_cqr") & invariance["K"].eq(10)
    ].pivot(index="fold", columns="rotation_seed", values="max_angle_deg").to_numpy()
    rows.append(
        {
            "method": "OTRR",
            "max_angle_deg": float(values.mean()),
            "max_angle_bootstrap_sd_deg": _fold_bootstrap_sd(
                values.mean(axis=1), seed=bootstrap_seed, repetitions=bootstrap_repetitions
            ),
        }
    )
    return pd.DataFrame(rows)


def build_table5(
    root: Path,
    prediction: pd.DataFrame,
    stability: pd.DataFrame,
    otrr_ci: pd.DataFrame,
    *,
    bootstrap_seed: int,
    bootstrap_repetitions: int,
) -> pd.DataFrame:
    baseline = pd.read_csv(root / "source/baselines/table5_baseline_update/table5_baseline_rows.csv")
    baseline_ci = dict(zip(baseline["method"], zip(baseline["average_95_contrast_ci_length"], baseline["ci_length_fold_bootstrap_sd"])))
    ci_map = {
        "LS": baseline_ci["ols"],
        "Radial Huber": baseline_ci["radial_huber"],
        "CW-Huber": baseline_ci["cw_huber"],
        "CW-ASM": baseline_ci["r_asm"],
        "CW-CQR": baseline_ci["cw_cqr"],
        "OTRR": (
            float(otrr_ci["average_95_ci_length"].mean()),
            _fold_bootstrap_sd(
                otrr_ci["average_95_ci_length"].to_numpy(),
                seed=bootstrap_seed,
                repetitions=bootstrap_repetitions,
            ),
        ),
    }
    equivariance = nhanes_equivariance().set_index("method")
    prediction = prediction.set_index("method")
    stability = stability.set_index("method")
    rows = []
    for method in METHODS:
        rows.append(
            {
                "method": method,
                "mae": float(prediction.loc[method, "mae"]),
                "mae_bootstrap_sd": float(prediction.loc[method, "mae_bootstrap_sd"]),
                "e_max_mean": float(equivariance.loc[method, "e_max_mean"]),
                "e_max_sd": float(equivariance.loc[method, "e_max_sd"]),
                "average_95_genus_contrast_ci_length": ci_map[method][0],
                "ci_length_fold_bootstrap_sd": ci_map[method][1],
                "stable_set_intersection": int(stability.loc[method, "intersection"]),
                "stable_set_union": int(stability.loc[method, "union"]),
                "stable_set_iou": float(stability.loc[method, "intersection_over_union"]),
            }
        )
    table = pd.DataFrame(rows)
    expected = load_frame("nhanes_summary.csv").set_index("method")
    for row in table.itertuples(index=False):
        for column in ["mae", "mae_bootstrap_sd", "average_95_genus_contrast_ci_length", "ci_length_fold_bootstrap_sd", "stable_set_intersection", "stable_set_union"]:
            if not np.isclose(getattr(row, column), expected.loc[row.method, column], rtol=1e-12, atol=1e-14):
                raise ValueError("retained NHANES summaries differ from the frozen paper version; recompute paired equivariance before combining new fits")
    return table


def _write_table5_tex(path: Path, table: pd.DataFrame) -> None:
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Method & MAE & $E_{\max}$ & 95\% CI length & Stable-set IoU \\",
        r"\hline",
    ]
    for row in table.itertuples(index=False):
        equivariance = equivariance_cell(row.e_max_mean, row.e_max_sd, threshold=1e-14)
        lines.append(
            f"{row.method} & {row.mae:.4f} ({row.mae_bootstrap_sd:.4f}) & {equivariance} & "
            f"{row.average_95_genus_contrast_ci_length:.4f} ({row.ci_length_fold_bootstrap_sd:.4f}) & "
            f"{row.stable_set_intersection}/{row.stable_set_union} \\\\"
        )
    lines.extend((r"\hline", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n")


def _write_pairwise_stability_tex(path: Path, pairwise: pd.DataFrame) -> None:
    pairs = (("helmert", "rotation_20260731"), ("helmert", "rotation_20260732"),
             ("rotation_20260731", "rotation_20260732"))
    lines = [
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Method & $H$ vs. $R_1$ & $H$ vs. $R_2$ & $R_1$ vs. $R_2$ \\",
        r"\hline",
    ]
    for method in METHODS:
        cells = []
        for left, right in pairs:
            row = pairwise[
                pairwise["method"].eq(method)
                & pairwise["representation_1"].eq(left)
                & pairwise["representation_2"].eq(right)
            ].iloc[0]
            cells.append(
                f"{int(row.intersection)}/{int(row.union)} "
                f"({100.0 * float(row.intersection_over_union):.1f}\\%)"
            )
        lines.append(f"{method} & " + " & ".join(cells) + r" \\")
    lines.extend((r"\hline", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(target, link.parent))


def _method_slug(display: str) -> str:
    return display.lower().replace(" ", "_").replace("-", "_")


def build_run_index(root: Path) -> None:
    runs = root / "runs"
    if runs.exists():
        shutil.rmtree(runs)
    main, rotated = _otrr_roots(root)
    baseline_root = root / "source/baselines/train_half_q01"
    cqr_basis = root / "source/cw_cqr_basis_inference"
    for fold in range(1, 6):
        base = runs / f"fold_{fold:02d}"
        for display, method in METHOD_IDS.items():
            _safe_link(
                base / "baselines" / _method_slug(display),
                baseline_root / f"fold_{fold:02d}" / method,
            )
        _safe_link(base / "cw_cqr_basis_fits", cqr_basis / f"fold_{fold:02d}")
        _safe_link(base / "otrr/helmert/members", main / f"original_fits/fold_{fold:02d}/cw_cqr")
        _safe_link(base / "otrr/helmert/inference", main / f"inference/fold_{fold:02d}/cw_cqr/K10")
        for seed in ROTATIONS:
            representation = f"rotation_{seed}"
            _safe_link(base / f"otrr/{representation}/members", main / f"rotation_fits/fold_{fold:02d}/cw_cqr/{representation}")
            _safe_link(base / f"otrr/{representation}/inference", rotated / f"artifacts/{representation}/fold_{fold:02d}")


def build_manifest(root: Path) -> pd.DataFrame:
    source_roots = {
        "baseline_fits": root / "source/baselines",
        "baseline_basis_inference": root / "source/baseline_basis_stability",
        "cw_cqr_basis_inference": root / "source/cw_cqr_basis_inference",
        "otrr_fits": root / "source/otrr" / OTRR_MAIN_DIR,
        "otrr_rotated_inference": root / "source/otrr" / OTRR_ROTATED_INFERENCE_DIR,
    }
    rows = []
    for role, source_root in source_roots.items():
        for path in sorted(source_root.resolve().rglob("*")):
            if path.is_file():
                rows.append(
                    {
                        "source_role": role,
                        "relative_path": str(path.relative_to(source_root.resolve())),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
    return pd.DataFrame(rows)


def finalize(root: Path, *, bootstrap_repetitions: int, bootstrap_seed: int) -> None:
    audit = validate_sources(root)
    for name in ("config", "metrics", "summaries", "tables", "provenance"):
        path = root / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)

    prediction, otrr_folds = build_prediction_summary(
        root, bootstrap_repetitions=bootstrap_repetitions, bootstrap_seed=bootstrap_seed
    )
    otrr_contrasts, otrr_ci = _otrr_contrast_outputs(root)
    stability, pairwise = build_stability_outputs(root)
    table5 = build_table5(
        root,
        prediction,
        stability,
        otrr_ci,
        bootstrap_seed=bootstrap_seed,
        bootstrap_repetitions=bootstrap_repetitions,
    )

    prediction.to_csv(root / "metrics/prediction_oof.csv", index=False)
    otrr_folds.to_csv(root / "metrics/otrr_prediction_by_fold.csv", index=False)
    otrr_contrasts.to_csv(root / "metrics/otrr_helmert_genus_contrasts.csv", index=False)
    pd.read_csv(root / "source/baseline_basis_stability/genus_vs_other_contrasts.csv").to_csv(
        root / "metrics/baseline_basis_genus_contrasts.csv", index=False
    )
    pd.read_csv(root / "source/cw_cqr_basis_inference/genus_vs_other_contrasts.csv").to_csv(
        root / "metrics/cw_cqr_basis_genus_contrasts.csv", index=False
    )
    _, rotated = _otrr_roots(root)
    pd.read_csv(rotated / "genus_vs_other_contrasts.csv").to_csv(
        root / "metrics/otrr_basis_genus_contrasts.csv", index=False
    )
    stability.to_csv(root / "summaries/stable_set_summary.csv", index=False)
    pairwise.to_csv(root / "summaries/pairwise_stability.csv", index=False)
    table5.to_csv(root / "tables/table5.csv", index=False)
    _write_table5_tex(root / "tables/table5.tex", table5)
    _write_pairwise_stability_tex(root / "tables/table_s2_3.tex", pairwise)

    prepared = _prepared(root, 1)
    pd.DataFrame(
        {
            "predictor_index": np.arange(1, len(prepared["predictor_names"]) + 1),
            "predictor": prepared["predictor_names"].astype(str),
        }
    ).to_csv(root / "tables/table_s2_2_predictors.csv", index=False)
    (root / "tables/table_s2_2_STATUS.md").write_text(
        "Table S2.2 contains the 43 retained NHANES predictors. The machine-readable "
        "predictor list is `table_s2_2_predictors.csv`; grouping and final LaTeX layout "
        "remain intentionally unresolved.\n",
        encoding="utf-8",
    )

    main, rotated = _otrr_roots(root)
    helmert_inference = pd.read_csv(main / "cw_cqr_only_summary/inference_summary.csv")
    helmert_inference = helmert_inference[
        ~helmert_inference["metric"].eq("trace_v_per_pd")
    ]
    helmert_inference.to_csv(root / "summaries/otrr_helmert_inference.csv", index=False)
    basis_inference = pd.read_csv(rotated / "representation_inference_metrics.csv")
    basis_inference = basis_inference.drop(columns=["trace_v_per_pd"], errors="ignore")
    basis_inference.to_csv(root / "summaries/otrr_basis_inference.csv", index=False)
    pd.read_csv(main / "cw_cqr_only_summary/invariance_metrics.csv").query(
        "method == 'otrr_cw_cqr' and K == 10"
    ).to_csv(root / "summaries/otrr_invariance.csv", index=False)

    config = {
        "experiment": "nhanes_p43_oral_microbiome_d7_final",
        "n": 8237,
        "p": 43,
        "d": 7,
        "composition_parts": list(COMPONENTS),
        "zero_replacement": "train_half_q01",
        "folds": 5,
        "split_seed": 20260726,
        "methods": list(METHODS),
        "otrr": {
            "initializer": "CW-CQR",
            "K": 10,
            "m_ref": 65536,
            "epsilon": 0.1,
            "steps": 500,
            "optimizer": "Adam",
            "learning_rate": 0.0001,
            "sinkhorn_updates_per_step": 1,
            "sinkhorn_warm_start": True,
            "backend": "dense_triton_float32",
            "reference_seeds": list(REFERENCE_SEEDS),
        },
        "inference": {
            "method": "plugin_covariance",
            "residuals": "estimated",
            "vectorization": "column-major vec(B)",
            "dual_tolerance": 0.001,
            "dual_max_iterations": 500,
            "dual_check_every": 5,
            "ensemble_rule": "average reference-wise IF, then form covariance",
        },
        "rotation_seeds": list(ROTATIONS),
        "bootstrap": {"repetitions": bootstrap_repetitions, "seed": bootstrap_seed},
        "survey_weights": False,
    }
    _write_json(root / "config/final_config.json", config)
    build_run_index(root)
    manifest = build_manifest(root)
    manifest.to_csv(root / "provenance/source_manifest.csv", index=False)
    audit.update(
        {
            "source_manifest_files": len(manifest),
            "source_manifest_bytes": int(manifest["bytes"].sum()),
            "paper_branch": "OTRR initialized by CW-CQR",
            "source_layout": "platform-neutral role-based source bundle",
        }
    )
    _write_json(root / "audit.json", audit)
    (root / "exact_commands.txt").write_text(
        "python run_real_data_experiment.py --config configs/main_real_data.json\n"
    )
    (root / "REPORT.md").write_text(
        "# Canonical NHANES p=43, d=7 results\n\n"
        "This directory is the paper-facing real-data result bundle. The formal response is "
        "the seven-dimensional Helmert ILR representation of an eight-part composition, and "
        "the formal OTRR estimator uses CW-CQR initialization, K=10, m_ref=65,536, and 500 Adam steps.\n\n"
        "Generated summaries are in `metrics/`, `summaries/`, and `tables/`. The immutable fitted "
        "states and inference artifacts are organized by role under `source/`; `runs/` provides "
        "a uniform fold-level index without duplicating large artifacts. The source manifest records "
        "every source file and SHA-256 digest. Internal historical labels `r_asm` are displayed as "
        "CW-ASM.\n\n"
        "Strict stable contrasts have the same sign and foldwise BH-adjusted q<0.05 in all five folds.\n"
    )
    (root / "COMPLETE.marker").write_text("NHANES_P43_D7_FINAL_COMPLETE\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    args = parser.parse_args()
    finalize(
        args.root.resolve(),
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
