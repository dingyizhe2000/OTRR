"""Prepared-fold and response-rotation protocol for NHANES CW-CQR."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from otrank.real_data.preprocessing import (
    fit_preprocessor,
    select_full_rank_columns,
    transform_response,
)
from otrank.real_data.nhanes_oral_baselines import load_analysis_data


P43_COLUMNS = (
    "age", "male", "income_poverty_ratio", "cycle_2011_2012", "race_1",
    "race_2", "race_4", "race_5", "education_1", "education_2", "education_3",
    "education_4", "marital_2", "marital_3", "marital_4", "marital_5",
    "marital_6", "bmi", "smoking_ever", "smoking_current", "alcohol_12plus_year",
    "alcohol_drinks_per_day", "diabetes", "borderline_diabetes", "hypertension",
    "hba1c", "log1p_diet_kcal", "log1p_diet_sugar_g", "log1p_diet_fiber_g",
    "log1p_diet_alcohol_g", "self_reported_gum_disease",
    "oral_health_rating_worse", "gum_disease_treatment", "loose_teeth", "bone_loss",
    "abnormal_tooth_recent", "floss_days_week", "natural_tooth_count", "implant_count",
    "root_fragment_count", "pocket_depth_mean", "attachment_loss_mean",
    "prescription_count",
)


def response_rotation(dimension: int, seed: int) -> np.ndarray:
    """Generate the fixed orthogonal response rotation used by the application."""

    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs[None, :]


def load_prepared_cw_cqr_jobs(
    prepared_dir: Path,
    out: Path,
    rotation_seeds: list[int],
    oof_target_file: Path | None = None,
) -> tuple[
    dict[int, dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    int,
    list[str],
    list[str],
]:
    """Load and validate exported p=43 folds and construct representation jobs."""

    fold_files = sorted(prepared_dir.glob("fold_[0-9][0-9].npz"))
    if not fold_files:
        raise FileNotFoundError(f"No prepared fold files found under {prepared_dir}")
    prepared: dict[int, dict[str, object]] = {}
    jobs: list[dict[str, object]] = []
    participant_ids: set[int] = set()
    predictor_names: list[str] | None = None
    response_names: list[str] | None = None
    authoritative_y: np.ndarray | None = None
    authoritative_fold: np.ndarray | None = None
    if oof_target_file is not None:
        with np.load(oof_target_file, allow_pickle=False) as oof:
            if "y_true" not in oof.files:
                raise ValueError(f"{oof_target_file} does not contain y_true")
            authoritative_y = np.asarray(oof["y_true"], dtype=np.float64)
            if "fold" in oof.files:
                authoritative_fold = np.asarray(oof["fold"], dtype=np.int64)

    for fold_file in fold_files:
        split = int(fold_file.stem.split("_")[-1])
        with np.load(fold_file, allow_pickle=False) as state:
            required = {"X_train", "X_test", "Y_train", "Y_test"}
            missing = required - set(state.files)
            if missing:
                raise ValueError(f"{fold_file} is missing arrays: {sorted(missing)}")
            x_train = np.asarray(state["X_train"], dtype=np.float64)
            x_test = np.asarray(state["X_test"], dtype=np.float64)
            y_train = np.asarray(state["Y_train"], dtype=np.float64)
            y_test = np.asarray(state["Y_test"], dtype=np.float64)
            train_idx = np.asarray(state["train_index"], dtype=np.int64)
            test_idx = np.asarray(state["test_index"], dtype=np.int64)
            current_predictors = [str(value) for value in state["predictor_names"]]
            current_responses = [str(value) for value in state["response_coordinate_names"]]
            if predictor_names is None:
                predictor_names = current_predictors
                response_names = current_responses
            elif predictor_names != current_predictors or response_names != current_responses:
                raise ValueError("Prepared folds do not share predictor/response ordering")
            if "train_participant_ids" in state.files:
                participant_ids.update(int(value) for value in state["train_participant_ids"])
                participant_ids.update(int(value) for value in state["test_participant_ids"])
        if authoritative_y is not None:
            if authoritative_y.ndim != 2 or authoritative_y.shape[0] <= max(
                int(train_idx.max()), int(test_idx.max())
            ):
                raise ValueError("Authoritative OOF y_true does not cover prepared indices")
            y_train = authoritative_y[train_idx]
            y_test = authoritative_y[test_idx]
            if authoritative_fold is not None and not np.all(
                authoritative_fold[test_idx] == split
            ):
                raise ValueError(f"OOF fold labels disagree for prepared fold {split}")
        if x_train.shape[1] != len(P43_COLUMNS) or predictor_names != list(P43_COLUMNS):
            raise ValueError("Prepared folds do not match the formal p=43 predictor profile")
        if not all(np.isfinite(value).all() for value in (x_train, x_test, y_train, y_test)):
            raise ValueError(f"Nonfinite prepared values in {fold_file}")

        prepared[split] = {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "zero_audit": {"source": "authoritative_prepared_fold"},
        }
        fold_dir = out / f"fold_{split:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fold_dir / "prepared.npz",
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            train_idx=train_idx,
            test_idx=test_idx,
        )
        jobs.append(
            {"split": split, "label": "original", "x": x_train, "y": y_train,
             "out": fold_dir / "original"}
        )
        for seed in rotation_seeds:
            q = response_rotation(y_train.shape[1], seed)
            jobs.append(
                {"split": split, "label": f"rotation_{seed}", "x": x_train,
                 "y": y_train @ q.T, "out": fold_dir / f"rotation_{seed}"}
            )

    metadata_path = prepared_dir / "fold_metadata.json"
    data_audit = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    data_audit["prepared_source"] = str(prepared_dir.resolve())
    if oof_target_file is not None:
        data_audit["response_override"] = str(oof_target_file.resolve()) + "::y_true"
    n_total = len(participant_ids) if participant_ids else max(
        len(value["train_idx"]) + len(value["test_idx"]) for value in prepared.values()
    )
    return prepared, jobs, data_audit, n_total, predictor_names or [], response_names or []


def prepare_raw_cw_cqr_jobs(
    data_root: Path,
    out: Path,
    *,
    min_reads: int,
    outer_folds: int,
    fold_limit: int | None,
    split_seed: int,
    zero_method: str,
    rotation_seeds: list[int],
) -> tuple[
    dict[int, dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    int,
    list[str],
    list[str],
]:
    """Build p=43 folds from source NHANES tables using the formal protocol."""

    predictors, response, _, strata, data_audit = load_analysis_data(
        data_root, min_reads=int(min_reads)
    )
    splitter = StratifiedKFold(
        n_splits=int(outer_folds), shuffle=True, random_state=int(split_seed)
    )
    splits = list(splitter.split(np.arange(len(predictors)), strata.to_numpy()))
    if fold_limit is not None:
        splits = splits[: int(fold_limit)]

    prepared: dict[int, dict[str, object]] = {}
    jobs: list[dict[str, object]] = []
    for split_zero, (train_idx, test_idx) in enumerate(splits):
        split = split_zero + 1
        x_train_frame, x_test_frame = fit_preprocessor(
            predictors.iloc[train_idx], predictors.iloc[test_idx], set(predictors.columns)
        )
        x_train_frame, x_test_frame = select_full_rank_columns(x_train_frame, x_test_frame)
        missing = sorted(set(P43_COLUMNS) - set(x_train_frame.columns))
        if missing:
            raise ValueError(f"p=43 profile columns missing in fold {split}: {missing}")
        x_train = x_train_frame.loc[:, P43_COLUMNS].to_numpy(dtype=np.float64)
        x_test = x_test_frame.loc[:, P43_COLUMNS].to_numpy(dtype=np.float64)
        y_train, y_test, zero_audit = transform_response(
            response.iloc[train_idx], response.iloc[test_idx], zero_method
        )
        prepared[split] = {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "zero_audit": zero_audit,
        }
        fold_dir = out / f"fold_{split:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fold_dir / "prepared.npz",
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            train_idx=train_idx,
            test_idx=test_idx,
        )
        jobs.append(
            {"split": split, "label": "original", "x": x_train, "y": y_train,
             "out": fold_dir / "original"}
        )
        for seed in rotation_seeds:
            q = response_rotation(y_train.shape[1], seed)
            jobs.append(
                {"split": split, "label": f"rotation_{seed}", "x": x_train,
                 "y": y_train @ q.T, "out": fold_dir / f"rotation_{seed}"}
            )
    return (
        prepared,
        jobs,
        data_audit,
        len(predictors),
        list(P43_COLUMNS),
        list(response.columns),
    )


__all__ = [
    "P43_COLUMNS",
    "load_prepared_cw_cqr_jobs",
    "prepare_raw_cw_cqr_jobs",
    "response_rotation",
]
