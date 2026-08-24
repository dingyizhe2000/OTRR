"""NHANES d=7 semi-synthetic invariance and Wald-inference experiment.

The experiment keeps the formal standardized NHANES predictor matrix fixed,
uses a sparse contrast template derived from the paper-facing OTRR fits, and
refits each baseline after independently rotating the response coordinates.
Artifacts are deliberately self-validating: every fit is bound to the config,
template, rotation bank, generated response, method, and source-code hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from otrank.project_layout import code_root
from scipy.stats import norm

from otrank.data.noise import PAPER_NOISES, make_generator, sample_noise
from otrank.estimators.baselines import (
    LinearModelFit,
    fit_coordinatewise_huber,
    fit_ols,
    fit_radial_huber,
)
from otrank.estimators.comparison import ASMConfig, fit_coordinatewise_asm_with_inference
from otrank.estimators.cw_cqr import benjamini_hochberg, cw_cqr_covariance
from otrank.estimators.cw_cqr_bridge import (
    CWCQRFitConfig,
    equally_spaced_quantiles,
    fit_cw_cqr,
)
from otrank.inference.asm import asm_inference_from_fit_state
from otrank.inference.linear import linear_model_influence
from otrank.real_data.preprocessing import _helmert_basis
from otrank.vectorization import COEFFICIENT_VECTORIZATION


SCHEMA_VERSION = 1
METHODS = ("LS", "Radial Huber", "CW-Huber", "CW-ASM", "CW-CQR")
NOISES = ("C",)
KAPPAS = (0.25, 0.5)
ROTATION_SEEDS = (2026081401, 2026081402)
REPRESENTATIONS = ("identity",) + tuple(f"rotation_{seed}" for seed in ROTATION_SEEDS)
GENERA = (
    "Actinomyces",
    "Rothia",
    "Prevotella 7",
    "Gemella",
    "Streptococcus",
    "Veillonella",
    "Haemophilus",
)
COMPONENTS = GENERA + ("Other",)
FORMAL_ROTATION_SEEDS = (20260731, 20260732)
FORMAL_REPRESENTATIONS = ("helmert",) + tuple(
    f"rotation_{seed}" for seed in FORMAL_ROTATION_SEEDS
)


@dataclass(frozen=True)
class ExperimentConfig:
    repetitions: int = 10
    workers: int = 4
    bh_level: float = 0.05
    derivative_ridge: float = 1e-10
    template_rule: str = "three_representation_strict_intersection"
    cqr_r_library: str | None = None
    cqr_quantile_levels: int = 20
    cqr_method: str = "ip"
    cqr_max_iterations: int = 1000
    cqr_tolerance: float = 1e-7
    asm_max_iterations: int = 65
    asm_alt_iterations: int = 1
    asm_package_version: str = "0.2.4"
    cqr_package_version: str = "1.2.1"
    vectorization: str = COEFFICIENT_VECTORIZATION
    intercept_convention: str = "mean_intercept_centered_slopes"
    covariance_convention: str = "V=Cov(sqrt(n)*(vec(Bhat)-vec(B0)))"
    dgp_orientation: str = "Y=X@(kappa*B_star).T+xi"
    rotation_convention: str = "Y_Q=Y@Q.T; B_back=Q.T@B_Q"
    source_data_seed: int = 202608140000


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required config field: {context}.{key}")
    return mapping[key]


def resolve_cqr_r_library(
    cli_value: str | None,
    *,
    environment: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve the optional CW-CQR R library without machine-specific paths."""

    env = os.environ if environment is None else environment
    for value in (cli_value, env.get("CQR_R_LIB"), env.get("R_LIBS_USER")):
        if value and str(value).strip():
            return str(Path(str(value)).expanduser())
    return None


def load_experiment_config(
    path: Path | str,
    *,
    repetitions: int | None = None,
    workers: int | None = None,
    r_library: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[ExperimentConfig, dict[str, Any], dict[str, Any]]:
    """Load and validate the authoritative semi-synthetic JSON config."""

    config_path = Path(path)
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load experiment config {config_path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("experiment config must be a JSON object")

    for key in (
        "experiment", "repetitions", "n", "p", "d", "methods", "noise",
        "kappa", "rotations", "template", "multiple_testing", "inference",
        "baselines", "otrr", "runtime", "source_data_seed",
    ):
        _required(document, key, "config")
    if document["experiment"] != "nhanes_p43_d7_semisynthetic_orthogonal_invariance":
        raise ValueError("unexpected experiment identifier")
    if (int(document["n"]), int(document["p"]), int(document["d"])) != (8237, 43, 7):
        raise ValueError("NHANES semi-synthetic config requires n=8237, p=43, d=7")
    if tuple(document["methods"]) != METHODS + ("OTRR",):
        raise ValueError(f"methods must be {list(METHODS + ('OTRR',))}")
    if tuple(document["noise"]) != NOISES:
        raise ValueError(f"noise must be {list(NOISES)}")
    if tuple(float(value) for value in document["kappa"]) != KAPPAS:
        raise ValueError(f"kappa must be {list(KAPPAS)}")

    rotations = document["rotations"]
    if not isinstance(rotations, dict) or rotations.get("identity") is not True:
        raise ValueError("rotations.identity must be true")
    if tuple(int(value) for value in _required(rotations, "haar_seeds", "rotations")) != ROTATION_SEEDS:
        raise ValueError(f"rotation seeds must be {list(ROTATION_SEEDS)}")

    testing = document["multiple_testing"]
    inference = document["inference"]
    template = document["template"]
    baselines = document["baselines"]
    otrr = document["otrr"]
    runtime = document["runtime"]
    for value, context in ((testing, "multiple_testing"), (inference, "inference"),
                           (template, "template"), (baselines, "baselines"),
                           (otrr, "otrr"), (runtime, "runtime")):
        if not isinstance(value, dict):
            raise ValueError(f"{context} must be an object")

    bh_level = float(_required(testing, "q", "multiple_testing"))
    if not 0.0 < bh_level < 1.0 or testing.get("procedure") != "Benjamini-Hochberg":
        raise ValueError("multiple_testing must specify Benjamini-Hochberg with q in (0,1)")
    if inference.get("vectorization") != COEFFICIENT_VECTORIZATION:
        raise ValueError(f"inference.vectorization must be {COEFFICIENT_VECTORIZATION!r}")
    if inference.get("covariance") != "analytic baseline influence-function/sandwich covariance":
        raise ValueError("unexpected inference covariance convention")
    if inference.get("intercept") != "mean intercept with centered slopes":
        raise ValueError("unexpected inference intercept convention")
    if int(_required(template, "expected_nonzero_contrasts", "template")) != 71:
        raise ValueError("template.expected_nonzero_contrasts must be 71")

    asm = _required(baselines, "cw_asm", "baselines")
    cqr = _required(baselines, "cw_cqr", "baselines")
    if not isinstance(asm, dict) or not isinstance(cqr, dict):
        raise ValueError("baselines.cw_asm and baselines.cw_cqr must be objects")
    reference_seeds = tuple(int(value) for value in _required(otrr, "reference_seeds", "otrr"))
    required_otrr = {
        "initializer": "CW-CQR", "K": 10, "m_ref": 65536, "epsilon": 0.1,
        "steps": 500, "learning_rate": 0.0001,
        "reference_convention": "same fixed unrotated cloud under all representations",
        "covariance": "plugin_covariance",
    }
    for key, expected in required_otrr.items():
        if otrr.get(key) != expected:
            raise ValueError(f"otrr.{key} must be {expected!r}")
    if reference_seeds != tuple(range(20260721, 20260731)):
        raise ValueError("otrr.reference_seeds must be 20260721,...,20260730")
    if runtime.get("multiprocessing") != "spawn" or int(runtime.get("blas_threads_per_worker", 0)) != 1:
        raise ValueError("runtime must use spawn and one BLAS thread per worker")

    final_repetitions = int(document["repetitions"] if repetitions is None else repetitions)
    final_workers = int(runtime.get("workers") if workers is None else workers)
    if final_repetitions < 1 or final_workers < 1:
        raise ValueError("repetitions and workers must be positive")
    resolved_library = resolve_cqr_r_library(r_library, environment=environment)
    overrides = {
        "repetitions": final_repetitions,
        "workers": final_workers,
        "r_library": resolved_library,
    }
    config = ExperimentConfig(
        repetitions=final_repetitions,
        workers=final_workers,
        bh_level=bh_level,
        derivative_ridge=float(_required(cqr, "derivative_ridge", "baselines.cw_cqr")),
        template_rule=str(_required(template, "support", "template")),
        cqr_r_library=resolved_library,
        cqr_quantile_levels=int(_required(cqr, "quantile_levels", "baselines.cw_cqr")),
        cqr_method=str(_required(cqr, "method", "baselines.cw_cqr")),
        cqr_max_iterations=int(_required(cqr, "max_iterations", "baselines.cw_cqr")),
        cqr_tolerance=float(_required(cqr, "tolerance", "baselines.cw_cqr")),
        asm_max_iterations=int(_required(asm, "max_iterations", "baselines.cw_asm")),
        asm_alt_iterations=int(_required(asm, "alternating_iterations", "baselines.cw_asm")),
        vectorization=str(inference["vectorization"]),
        intercept_convention=str(inference["intercept"]),
        covariance_convention=str(inference["covariance"]),
        source_data_seed=int(document["source_data_seed"]),
    )
    return config, document, overrides


@dataclass(frozen=True)
class TaskSpec:
    repetition: int
    noise: str
    kappa: float

    @property
    def task_id(self) -> str:
        return f"rep{self.repetition:03d}_noise{self.noise}_kappa{self.kappa:g}"


def canonical_json(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def haar_rotation(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    raw = rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(raw)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs[None, :]


def formal_response_rotation(dimension: int, seed: int) -> np.ndarray:
    """Reproduce the NumPy QR convention used by the d=7 formal runner."""

    return haar_rotation(dimension, seed)


def build_rotation_bank() -> tuple[np.ndarray, np.ndarray]:
    rotations = [np.eye(7, dtype=np.float64)]
    rotations.extend(haar_rotation(7, seed) for seed in ROTATION_SEEDS)
    bank = np.stack(rotations)
    errors = np.max(np.abs(bank @ np.swapaxes(bank, 1, 2) - np.eye(7)), axis=(1, 2))
    if float(errors.max()) > 1e-12:
        raise ValueError("rotation bank failed orthogonality check")
    return bank, errors


def contrast_matrix(components: int = 8) -> tuple[np.ndarray, np.ndarray]:
    h = _helmert_basis(components).T
    raw = np.eye(components - 1, components) - np.eye(components)[-1][None, :]
    a_h = raw @ h
    if h.shape != (8, 7) or a_h.shape != (7, 7):
        raise ValueError("unexpected Helmert/contrast shape")
    if float(np.max(np.abs(h.T @ h - np.eye(7)))) > 1e-12:
        raise ValueError("Helmert basis is not orthonormal")
    if float(np.max(np.abs(h.T @ np.ones(8)))) > 1e-12:
        raise ValueError("Helmert basis is not orthogonal to the closure direction")
    return h, a_h


def _stable_contrast_intersection(contrast_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    values = pd.read_csv(contrast_path)
    required = {
        "fold",
        "representation",
        "predictor_index",
        "genus_index",
        "estimate",
        "bh_q_value",
    }
    missing = required.difference(values.columns)
    if missing:
        raise ValueError(f"formal contrast table is missing {sorted(missing)}")
    sets: dict[str, set[tuple[int, int]]] = {}
    rows: list[dict[str, Any]] = []
    for representation in FORMAL_REPRESENTATIONS:
        selected: set[tuple[int, int]] = set()
        subset = values[values["representation"] == representation]
        for (predictor, genus), group in subset.groupby(["predictor_index", "genus_index"]):
            signs = np.sign(group["estimate"].to_numpy(dtype=np.float64))
            stable = (
                len(group) == 5
                and bool(np.all(group["bh_q_value"].to_numpy(dtype=np.float64) < 0.05))
                and bool(np.all(signs == signs[0]))
                and signs[0] != 0.0
            )
            if stable:
                selected.add((int(genus), int(predictor)))
        sets[representation] = selected
    intersection = set.intersection(*(sets[name] for name in FORMAL_REPRESENTATIONS))
    mask = np.zeros((7, 43), dtype=bool)
    for genus, predictor in intersection:
        mask[genus, predictor] = True
    for name in FORMAL_REPRESENTATIONS:
        rows.append(
            {
                "representation": name,
                "stable_count": len(sets[name]),
                "intersection_count": len(intersection),
                "union_count": len(set.union(*(sets[key] for key in FORMAL_REPRESENTATIONS))),
            }
        )
    if int(mask.sum()) != 71:
        raise ValueError(f"expected 71 three-representation stable contrasts, found {mask.sum()}")
    return mask, pd.DataFrame(rows)


def _load_otrr_template_slope(formal_root: Path) -> tuple[np.ndarray, pd.DataFrame]:
    main = formal_root / "source/otrr/formal_cw_cqr_K10"
    coefficient_rows: list[dict[str, Any]] = []
    mapped: list[np.ndarray] = []
    for fold in range(1, 6):
        for representation in FORMAL_REPRESENTATIONS:
            if representation == "helmert":
                rotation = np.eye(7)
                pattern = main / f"original_fits/fold_{fold:02d}/cw_cqr"
            else:
                seed = int(representation.removeprefix("rotation_"))
                rotation = formal_response_rotation(7, seed)
                pattern = main / f"rotation_fits/fold_{fold:02d}/cw_cqr/{representation}"
            members = []
            member_hashes = []
            for reference, seed in enumerate(range(20260721, 20260731), 1):
                path = pattern / f"reference_{reference:02d}_seed{seed}/iteration_500/state.pt"
                if not path.is_file():
                    raise FileNotFoundError(path)
                state = torch.load(path, map_location="cpu", weights_only=False)
                coef = np.asarray(state["coef"], dtype=np.float64)
                if coef.shape != (7, 43) or not np.isfinite(coef).all():
                    raise ValueError(f"invalid OTRR coefficient state: {path}")
                members.append(coef)
                member_hashes.append(sha256_file(path))
            ensemble = np.mean(np.stack(members), axis=0)
            mapped_back = rotation.T @ ensemble
            mapped.append(mapped_back)
            coefficient_rows.append(
                {
                    "fold": fold,
                    "representation": representation,
                    "member_count": 10,
                    "ensemble_sha256": hash_array(ensemble),
                    "mapped_back_sha256": hash_array(mapped_back),
                    "member_state_sha256": sha256_bytes(canonical_json(member_hashes).encode("utf-8")),
                }
            )
    return np.mean(np.stack(mapped), axis=0), pd.DataFrame(coefficient_rows)


def prepare_template(project_root: Path, output_root: Path) -> dict[str, Any]:
    source_root = code_root(project_root)
    formal_root = project_root / "results/real_data/nhanes_p43_final"
    x_path = (
        project_root
        / "results/real_data/nhanes_p43_final/supplement/"
        "semisynthetic_invariance/baselines/prepared/template.npz"
    )
    contrast_path = formal_root / "metrics/otrr_basis_genus_contrasts.csv"
    with np.load(x_path, allow_pickle=False) as prepared:
        x = np.asarray(prepared["X"], dtype=np.float64)
        participant_ids = np.asarray(prepared["participant_ids"])
        predictor_names = np.asarray(prepared["predictor_names"]).astype(str)
    if x.shape != (8237, 43) or len(np.unique(participant_ids)) != 8237:
        raise ValueError("formal full-sample design has unexpected shape or duplicate IDs")
    if not np.isfinite(x).all() or float(np.max(np.abs(x.mean(axis=0)))) > 1e-5:
        raise ValueError("formal full-sample design is not finite and globally centered")

    h, a_h = contrast_matrix(8)
    mask, stability = _stable_contrast_intersection(contrast_path)
    b_otrr, fit_manifest = _load_otrr_template_slope(formal_root)
    theta_raw = a_h @ b_otrr
    theta_star = np.where(mask, theta_raw, 0.0)
    b_star = np.linalg.solve(a_h, theta_star)
    reconstruction = a_h @ b_star
    zero_error = float(np.max(np.abs(reconstruction[~mask])))
    kept_error = float(np.max(np.abs(reconstruction[mask] - theta_raw[mask])))
    if zero_error > 1e-12 or kept_error > 1e-10:
        raise ValueError("sparse contrast template failed exact reconstruction")

    rotations, rotation_errors = build_rotation_bank()
    prepared_root = output_root / "prepared"
    atomic_npz(
        prepared_root / "template.npz",
        X=x,
        participant_ids=participant_ids,
        predictor_names=predictor_names,
        components=np.asarray(COMPONENTS),
        genera=np.asarray(GENERA),
        H=h,
        A_H=a_h,
        B_otrr=b_otrr,
        Theta_raw=theta_raw,
        stable_mask=mask,
        Theta_star=theta_star,
        B_star=b_star,
    )
    atomic_npz(
        prepared_root / "rotation_bank.npz",
        rotations=rotations,
        names=np.asarray(REPRESENTATIONS),
        seeds=np.asarray((0,) + ROTATION_SEEDS, dtype=np.int64),
        orthogonality_errors=rotation_errors,
    )
    stability.to_csv(prepared_root / "formal_stability_audit.csv", index=False)
    fit_manifest.to_csv(prepared_root / "formal_otrr_fit_manifest.csv", index=False)
    pd.DataFrame(
        [
            {
                "genus_index": genus,
                "genus": GENERA[genus],
                "predictor_index": predictor,
                "predictor": predictor_names[predictor],
                "theta_raw": theta_raw[genus, predictor],
                "theta_star": theta_star[genus, predictor],
            }
            for genus, predictor in zip(*np.where(mask), strict=True)
        ]
    ).to_csv(prepared_root / "template_nonzero_contrasts.csv", index=False)
    source_code_paths = (
        Path(__file__).resolve(),
        source_root / "run_nhanes_semisynthetic_invariance.py",
        source_root / "src/otrank/data/noise.py",
        source_root / "src/otrank/estimators/baselines.py",
        source_root / "src/otrank/estimators/comparison.py",
        source_root / "src/otrank/estimators/cw_cqr.py",
        source_root / "src/otrank/estimators/cw_cqr_bridge.py",
        source_root / "src/otrank/estimators/fit_cw_cqr.R",
        source_root / "src/otrank/inference/linear.py",
        source_root / "src/otrank/inference/asm.py",
        source_root / "src/otrank/vectorization.py",
    )
    source_code_sha256 = {str(path): sha256_file(path) for path in source_code_paths}
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "template_rule": "mean of 15 mapped-back fold/representation K=10 OTRR slopes, then retain the 71 contrasts in the three-representation strict-stable intersection",
        "warning": "No separate full-sample d=7 OTRR fit is available; the template is a transparent aggregate of formal fold fits.",
        "n": 8237,
        "p": 43,
        "d": 7,
        "stable_contrast_count": int(mask.sum()),
        "exact_zero_max_abs": zero_error,
        "kept_reconstruction_max_abs": kept_error,
        "source_files": {
            "full_sample_design": {"path": str(x_path), "sha256": sha256_file(x_path)},
            "formal_contrasts": {"path": str(contrast_path), "sha256": sha256_file(contrast_path)},
        },
        "array_sha256": {
            "X": hash_array(x),
            "H": hash_array(h),
            "A_H": hash_array(a_h),
            "B_otrr": hash_array(b_otrr),
            "Theta_star": hash_array(theta_star),
            "B_star": hash_array(b_star),
            "rotation_bank": hash_array(rotations),
        },
        "source_code_sha256": source_code_sha256,
    }
    atomic_json(prepared_root / "template_provenance.json", provenance)
    return provenance


def config_payload(
    config: ExperimentConfig,
    template_provenance: Mapping[str, Any],
    *,
    source_config: Mapping[str, Any] | None = None,
    runtime_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scientific_config = asdict(config)
    # Worker count is operational rather than scientific. Excluding it from the
    # hash permits a verified run to resume with fewer workers after a reboot.
    scientific_config.pop("workers")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "nhanes_p43_d7_semisynthetic_orthogonal_invariance",
        "n": 8237,
        "p": 43,
        "d": 7,
        "methods": list(METHODS),
        "noises": list(NOISES),
        "kappas": list(KAPPAS),
        "representations": list(REPRESENTATIONS),
        "rotation_seeds": list(ROTATION_SEEDS),
        "formal_noise_specs": {label: asdict(PAPER_NOISES[label]) for label in NOISES},
        "config": scientific_config,
        "runtime_workers_requested": int(config.workers),
        "template_array_sha256": dict(template_provenance["array_sha256"]),
        "source_code_sha256": dict(template_provenance["source_code_sha256"]),
        "source_config": dict(source_config or {}),
        "runtime_overrides": dict(runtime_overrides or {}),
    }
    hashed = dict(payload)
    hashed.pop("runtime_workers_requested")
    if "runtime_overrides" in hashed:
        hashed["runtime_overrides"] = {
            key: value
            for key, value in hashed["runtime_overrides"].items()
            if key != "workers"
        }
    payload["config_sha256"] = sha256_bytes(canonical_json(hashed).encode("utf-8"))
    return payload


def dgp_seed(config: ExperimentConfig, task: TaskSpec) -> int:
    noise_index = NOISES.index(task.noise)
    kappa_index = KAPPAS.index(float(task.kappa))
    return int(config.source_data_seed + task.repetition * 100 + noise_index * 10 + kappa_index)


def generate_response(
    x: np.ndarray,
    b_star: np.ndarray,
    task: TaskSpec,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    seed = dgp_seed(config, task)
    generator = make_generator(seed, "cpu")
    noise = sample_noise(
        len(x),
        b_star.shape[0],
        PAPER_NOISES[task.noise],
        generator,
        device="cpu",
        dtype=torch.float64,
    ).numpy()
    y = x @ (float(task.kappa) * b_star).T + noise
    return y, noise, seed


def _linear_fit_and_covariance(method: str, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    xt = torch.as_tensor(x, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64)
    if method == "LS":
        fit = fit_ols(xt, yt)
        result = linear_model_influence(x, y, fit, method="ols", intercept_mode="mean")
        return fit.coef.numpy(), result.covariance, result.diagnostics
    if method == "Radial Huber":
        fit = fit_radial_huber(xt, yt, intercept_mode="center")
        result = linear_model_influence(x, y, fit, method="radial_huber", intercept_mode="mean")
        return fit.coef.numpy(), result.covariance, result.diagnostics
    if method == "CW-Huber":
        fit = fit_coordinatewise_huber(xt, yt, intercept_mode="center")
        result = linear_model_influence(x, y, fit, method="coordinatewise_huber", intercept_mode="mean")
        return fit.coef.numpy(), result.covariance, result.diagnostics
    raise ValueError(method)


def _asm_fit_and_covariance(x: np.ndarray, y: np.ndarray, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fit = fit_coordinatewise_asm_with_inference(
        torch.as_tensor(x, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        ASMConfig(
            betapilot="OLS",
            symmetric=False,
            alt_iter=config.asm_alt_iterations,
            max_iter=config.asm_max_iterations,
            intercept_selection="mean",
            center_xy=True,
            response_workers=1,
            numeric_threads=1,
        ),
    )
    inference = asm_inference_from_fit_state(x, fit, center_x=True)
    diagnostics = dict(inference.diagnostics)
    diagnostics["package_version"] = fit.package_version
    return fit.coef.numpy(), inference.joint_covariance, diagnostics


def _cqr_fit_and_covariance(
    x: np.ndarray,
    y: np.ndarray,
    fit_dir: Path,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    tau = equally_spaced_quantiles(config.cqr_quantile_levels)
    fit = fit_cw_cqr(
        x,
        y,
        fit_dir,
        config=CWCQRFitConfig(
            tau=tuple(tau),
            method=config.cqr_method,
            max_iterations=config.cqr_max_iterations,
            tolerance=config.cqr_tolerance,
            r_library=config.cqr_r_library,
        ),
    )
    inference = cw_cqr_covariance(
        x,
        y,
        fit.beta,
        fit.quantile_intercepts,
        tau,
        derivative_ridge=config.derivative_ridge,
    )
    diagnostics = {
        "package_version": config.cqr_package_version,
        "fit_config_sha256": fit.config_hash,
        "fit_wall_sec": fit.wall_sec,
        "fit_resumed": fit.resumed,
        "bread_condition_numbers": inference.bread_condition_numbers.tolist(),
        "bread_ranks": inference.bread_ranks.tolist(),
    }
    return fit.beta, inference.covariance, diagnostics


def map_back(coefficient: np.ndarray, covariance: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d, p = coefficient.shape
    operator = np.kron(np.eye(p), rotation.T)
    mapped_coefficient = rotation.T @ coefficient
    mapped_covariance = operator @ covariance @ operator.T
    mapped_covariance = 0.5 * (mapped_covariance + mapped_covariance.T)
    if mapped_covariance.shape != (d * p, d * p):
        raise ValueError("mapped covariance has wrong shape")
    return mapped_coefficient, mapped_covariance


def contrast_statistics(
    coefficient: np.ndarray,
    covariance: np.ndarray,
    a_h: np.ndarray,
    n: int,
    bh_level: float,
) -> dict[str, np.ndarray]:
    d, p = coefficient.shape
    effects = a_h @ coefficient
    variances = np.empty((d, p), dtype=np.float64)
    for predictor in range(p):
        indices = slice(predictor * d, (predictor + 1) * d)
        block = covariance[indices, indices]
        variances[:, predictor] = np.einsum("gi,ij,gj->g", a_h, block, a_h)
    se = np.sqrt(np.clip(variances, 0.0, np.inf) / n)
    z = np.divide(effects, se, out=np.zeros_like(effects), where=se > 0.0)
    p_values = 2.0 * norm.sf(np.abs(z))
    q_values = benjamini_hochberg(p_values)
    selected = q_values < float(bh_level)
    return {
        "effects": effects,
        "standard_errors": se,
        "p_values": p_values,
        "q_values": q_values,
        "selected": selected,
    }


def method_artifact_path(output_root: Path, task: TaskSpec, representation: str, method: str) -> Path:
    method_slug = method.lower().replace(" ", "_").replace("-", "_")
    return output_root / "tasks" / task.task_id / representation / f"{method_slug}.npz"


def _validate_method_artifact(
    path: Path,
    *,
    config_sha256: str,
    template_sha256: str,
    rotation_sha256: str,
    y_sha256: str,
    task: TaskSpec,
    representation: str,
    method: str,
) -> bool:
    if not path.is_file():
        return False
    with np.load(path, allow_pickle=False) as state:
        expected = {
            "schema_version": str(SCHEMA_VERSION),
            "config_sha256": config_sha256,
            "template_sha256": template_sha256,
            "rotation_sha256": rotation_sha256,
            "y_sha256": y_sha256,
            "task_id": task.task_id,
            "representation": representation,
            "method": method,
            "vectorization": COEFFICIENT_VECTORIZATION,
        }
        for key, value in expected.items():
            if key not in state or str(state[key].item()) != value:
                raise ValueError(f"artifact metadata mismatch for {path}: {key}")
        required = ("coefficient", "covariance", "effects", "standard_errors", "p_values", "q_values", "selected")
        for key in required:
            if key not in state or not np.isfinite(state[key]).all():
                raise ValueError(f"artifact missing/non-finite {key}: {path}")
        coefficient = state["coefficient"]
        covariance = state["covariance"]
        if coefficient.shape != (7, 43) or covariance.shape != (301, 301):
            raise ValueError(f"artifact shape mismatch: {path}")
        if float(np.max(np.abs(covariance - covariance.T))) > 1e-8:
            raise ValueError(f"artifact covariance is not symmetric: {path}")
    return True


def run_task(
    project_root: Path,
    output_root: Path,
    task: TaskSpec,
    config: ExperimentConfig,
    config_sha256: str,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    with np.load(output_root / "prepared/template.npz", allow_pickle=False) as template:
        x = np.asarray(template["X"], dtype=np.float64)
        b_star = np.asarray(template["B_star"], dtype=np.float64)
        a_h = np.asarray(template["A_H"], dtype=np.float64)
    with np.load(output_root / "prepared/rotation_bank.npz", allow_pickle=False) as bank:
        rotations = np.asarray(bank["rotations"], dtype=np.float64)
        names = tuple(bank["names"].astype(str))
    template_sha256 = hash_array(b_star)
    y, noise, seed = generate_response(x, b_star, task, config)
    y_sha256 = hash_array(y)
    noise_sha256 = hash_array(noise)
    task_started = time.perf_counter()
    fit_count = 0
    for representation, rotation in zip(names, rotations, strict=True):
        y_rotated = y @ rotation.T
        rotation_sha256 = hash_array(rotation)
        for method in METHODS:
            artifact = method_artifact_path(output_root, task, representation, method)
            if _validate_method_artifact(
                artifact,
                config_sha256=config_sha256,
                template_sha256=template_sha256,
                rotation_sha256=rotation_sha256,
                y_sha256=y_sha256,
                task=task,
                representation=representation,
                method=method,
            ):
                fit_count += 1
                continue
            started = time.perf_counter()
            if method in {"LS", "Radial Huber", "CW-Huber"}:
                coefficient, covariance, diagnostics = _linear_fit_and_covariance(method, x, y_rotated)
            elif method == "CW-ASM":
                coefficient, covariance, diagnostics = _asm_fit_and_covariance(x, y_rotated, config)
            elif method == "CW-CQR":
                coefficient, covariance, diagnostics = _cqr_fit_and_covariance(
                    x,
                    y_rotated,
                    artifact.parent / "cw_cqr_fit",
                    config,
                )
            else:
                raise ValueError(method)
            mapped_coefficient, mapped_covariance = map_back(coefficient, covariance, rotation)
            statistics = contrast_statistics(mapped_coefficient, mapped_covariance, a_h, len(x), config.bh_level)
            wall = time.perf_counter() - started
            atomic_npz(
                artifact,
                schema_version=np.asarray(SCHEMA_VERSION),
                config_sha256=np.asarray(config_sha256),
                template_sha256=np.asarray(template_sha256),
                rotation_sha256=np.asarray(rotation_sha256),
                y_sha256=np.asarray(y_sha256),
                noise_sha256=np.asarray(noise_sha256),
                task_id=np.asarray(task.task_id),
                repetition=np.asarray(task.repetition),
                noise=np.asarray(task.noise),
                kappa=np.asarray(task.kappa),
                dgp_seed=np.asarray(seed),
                representation=np.asarray(representation),
                method=np.asarray(method),
                vectorization=np.asarray(COEFFICIENT_VECTORIZATION),
                coefficient=mapped_coefficient,
                covariance=mapped_covariance,
                covariance_sha256=np.asarray(hash_array(mapped_covariance)),
                effects=statistics["effects"],
                standard_errors=statistics["standard_errors"],
                p_values=statistics["p_values"],
                q_values=statistics["q_values"],
                selected=statistics["selected"],
                fit_wall_sec=np.asarray(wall),
                diagnostics_json=np.asarray(json.dumps(diagnostics, sort_keys=True)),
            )
            fit_count += 1
    task_manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task.task_id,
        "config_sha256": config_sha256,
        "template_sha256": template_sha256,
        "y_sha256": y_sha256,
        "noise_sha256": noise_sha256,
        "dgp_seed": seed,
        "expected_fit_count": len(REPRESENTATIONS) * len(METHODS),
        "complete_fit_count": fit_count,
        "wall_sec": time.perf_counter() - task_started,
        "status": "complete" if fit_count == len(REPRESENTATIONS) * len(METHODS) else "incomplete",
    }
    if task_manifest["status"] != "complete":
        raise RuntimeError(f"incomplete task: {task.task_id}")
    atomic_json(output_root / "tasks" / task.task_id / "COMPLETE.json", task_manifest)
    return task_manifest


def iter_tasks(repetitions: Iterable[int]) -> Iterable[TaskSpec]:
    for repetition in repetitions:
        for noise in NOISES:
            for kappa in KAPPAS:
                yield TaskSpec(int(repetition), noise, float(kappa))


def summarize(output_root: Path, config_sha256: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    with np.load(output_root / "prepared/template.npz", allow_pickle=False) as template:
        signal_mask = np.asarray(template["stable_mask"], dtype=bool)
    rows: list[dict[str, Any]] = []
    set_rows: list[dict[str, Any]] = []
    for complete in sorted((output_root / "tasks").glob("*/COMPLETE.json")):
        task_root = complete.parent
        task_meta = json.loads(complete.read_text(encoding="utf-8"))
        if task_meta.get("config_sha256") != config_sha256 or task_meta.get("status") != "complete":
            raise ValueError(f"invalid task completion metadata: {complete}")
        task_id = str(task_meta["task_id"])
        selected_by_method: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
        for representation in REPRESENTATIONS:
            for method in METHODS:
                path = method_artifact_path(
                    output_root,
                    TaskSpec(
                        int(task_id.split("_")[0].removeprefix("rep")),
                        task_id.split("_")[1].removeprefix("noise"),
                        float(task_id.split("_")[2].removeprefix("kappa")),
                    ),
                    representation,
                    method,
                )
                with np.load(path, allow_pickle=False) as state:
                    selected = np.asarray(state["selected"], dtype=bool)
                    repetition = int(state["repetition"])
                    noise = str(state["noise"].item())
                    kappa = float(state["kappa"])
                    wall = float(state["fit_wall_sec"])
                truth = signal_mask if kappa > 0.0 else np.zeros_like(signal_mask)
                tp = int(np.sum(selected & truth))
                fp = int(np.sum(selected & ~truth))
                fn = int(np.sum(~selected & truth))
                tn = int(np.sum(~selected & ~truth))
                rows.append(
                    {
                        "repetition": repetition,
                        "noise": noise,
                        "kappa": kappa,
                        "representation": representation,
                        "method": method,
                        "selected": int(selected.sum()),
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tn": tn,
                        "fdr": fp / max(int(selected.sum()), 1),
                        "tpr": tp / max(int(truth.sum()), 1),
                        "marginal_size": fp / max(int((~truth).sum()), 1),
                        "fit_wall_sec": wall,
                    }
                )
                selected_by_method[method].append(selected)
        for method, selections in selected_by_method.items():
            intersection = np.logical_and.reduce(selections)
            union = np.logical_or.reduce(selections)
            set_rows.append(
                {
                    "repetition": repetition,
                    "noise": noise,
                    "kappa": kappa,
                    "method": method,
                    "intersection_count": int(intersection.sum()),
                    "union_count": int(union.sum()),
                    "iou": float(intersection.sum() / union.sum()) if union.any() else 1.0,
                    "selection_count_min": int(min(x.sum() for x in selections)),
                    "selection_count_max": int(max(x.sum() for x in selections)),
                }
            )
    metrics = pd.DataFrame(rows)
    sets = pd.DataFrame(set_rows)
    summary_root = output_root / "summaries"
    summary_root.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(summary_root / "rotation_level_metrics.csv", index=False)
    sets.to_csv(summary_root / "set_stability_metrics.csv", index=False)
    if not metrics.empty:
        summary = (
            metrics.groupby(["noise", "kappa", "method"], as_index=False)
            .agg(
                repetitions=("repetition", "nunique"),
                selected_mean=("selected", "mean"),
                fdr_mean=("fdr", "mean"),
                tpr_mean=("tpr", "mean"),
                marginal_size_mean=("marginal_size", "mean"),
                tpr_rotation_min=("tpr", "min"),
                tpr_rotation_max=("tpr", "max"),
                fit_wall_sec_mean=("fit_wall_sec", "mean"),
            )
        )
        summary.to_csv(summary_root / "summary_by_noise_kappa_method.csv", index=False)
    if not sets.empty:
        sets.groupby(["noise", "kappa", "method"], as_index=False).agg(
            repetitions=("repetition", "nunique"),
            iou_mean=("iou", "mean"),
            iou_sd=("iou", "std"),
            intersection_mean=("intersection_count", "mean"),
            union_mean=("union_count", "mean"),
        ).to_csv(summary_root / "summary_set_stability.csv", index=False)
    return metrics, sets


__all__ = [
    "ExperimentConfig",
    "load_experiment_config",
    "resolve_cqr_r_library",
    "TaskSpec",
    "METHODS",
    "NOISES",
    "KAPPAS",
    "REPRESENTATIONS",
    "prepare_template",
    "config_payload",
    "iter_tasks",
    "run_task",
    "summarize",
    "contrast_matrix",
    "build_rotation_bank",
    "map_back",
    "contrast_statistics",
]
