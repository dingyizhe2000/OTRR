"""Strict metadata contracts for persisted OT-rank inference artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from otrank.vectorization import COEFFICIENT_VECTORIZATION, require_vectorization


PLUGIN_COVARIANCE_ARTIFACT_SCHEMA_VERSION = 5
PLUGIN_COVARIANCE_AGGREGATE_SCHEMA_VERSION = 1
PLUGIN_COVARIANCE_AGGREGATE_KIND = "plugin_covariance_aggregate"
PLUGIN_COVARIANCE_METHOD = "plugin_covariance"
PLUGIN_DUAL_DEFINITION = "reconstructed_at_final_coefficient_to_tolerance"
PLUGIN_COVARIANCE_BACKEND = "numpy_dense_linear_algebra"
PLUGIN_RANK_JACOBIAN_BACKEND = "chunked_pytorch"

_CONFIG_KEYS = (
    "artifact_schema_version",
    "inference_method",
    "vectorization",
    "coefficient_shape",
    "dual_definition",
    "dual_tolerance",
    "dual_max_iters",
    "dual_check_every",
    "dual_backend_requested",
    "dual_triton_block_m",
    "require_dual_convergence",
    "covariance_backend",
    "rank_jacobian_backend",
    "chunk_size",
    "derivative_ridge",
    "device_type",
    "compute_dtype",
    "intercept_mode",
    "n_train",
    "predictor_dimension",
    "response_dimension",
    "k_references",
    "reference_seeds",
    "m_ref",
    "epsilon",
    "fitted_sinkhorn_iters",
    "inference_code_sha256",
    "fit_inventory_sha256",
    "fit_inventory_reference_count",
    "fit_inventory_repetitions",
)

_REFERENCE_PROVENANCE_SHA_FIELDS = (
    "fit_state_sha256_by_reference",
    "rotated_state_sha256_by_reference",
    "job_config_sha256_by_reference",
    "trace_sha256_by_reference",
    "scientific_job_sha256_by_reference",
    "contribution_npz_sha256_by_reference",
    "reference_provenance_sha256_by_reference",
)

_DATA_PROVENANCE_SHA_FIELDS = (
    "x_train_sha256",
    "y_train_sha256",
    "truth_sha256",
    "rotation_sha256",
    "data_provenance_sha256",
)


def _plain_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _plain_value(value.item())
        return [_plain_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain_value(value.item())
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    return value


def _boolean(value: Any, *, field: str) -> bool:
    value = _plain_value(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ValueError(f"{field} must be an explicit boolean, got {value!r}")


def _coerce_like(value: Any, expected: Any, *, field: str) -> Any:
    """Coerce NPZ scalars and CSV strings to the canonical config type."""

    value = _plain_value(value)
    if isinstance(expected, bool):
        return _boolean(value, field=field)
    if isinstance(expected, int) and not isinstance(expected, bool):
        return int(value)
    if isinstance(expected, float):
        return float(value)
    if isinstance(expected, list):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"artifact field {field!r} is not a JSON list") from exc
        if not isinstance(value, list):
            raise ValueError(f"artifact field {field!r} is not a list")
        return [_coerce_like(item, template, field=field) for item, template in zip(value, expected)] \
            if len(value) == len(expected) else value
    return str(value) if isinstance(expected, str) else value


def plugin_covariance_config(
    *,
    dual_tolerance: float,
    dual_max_iters: int,
    dual_check_every: int,
    dual_backend_requested: str,
    dual_triton_block_m: int,
    chunk_size: int,
    derivative_ridge: float,
    device_type: str,
    n_train: int,
    predictor_dimension: int,
    response_dimension: int,
    k_references: int,
    reference_seeds: Sequence[int],
    m_ref: int,
    epsilon: float,
    fitted_sinkhorn_iters: int,
    inference_code_sha256: str,
    fit_inventory_sha256: str,
    fit_inventory_reference_count: int,
    fit_inventory_repetitions: Sequence[int],
    require_dual_convergence: bool = True,
    compute_dtype: str = "float32",
    intercept_mode: str = "zero",
) -> dict[str, Any]:
    """Return the complete, JSON-safe configuration defining one IF run."""

    config = {
        "artifact_schema_version": PLUGIN_COVARIANCE_ARTIFACT_SCHEMA_VERSION,
        "inference_method": PLUGIN_COVARIANCE_METHOD,
        "vectorization": COEFFICIENT_VECTORIZATION,
        "coefficient_shape": "(response_dimension, predictor_dimension)",
        "dual_definition": PLUGIN_DUAL_DEFINITION,
        "dual_tolerance": float(dual_tolerance),
        "dual_max_iters": int(dual_max_iters),
        "dual_check_every": int(dual_check_every),
        "dual_backend_requested": str(dual_backend_requested),
        "dual_triton_block_m": int(dual_triton_block_m),
        "require_dual_convergence": bool(require_dual_convergence),
        "covariance_backend": PLUGIN_COVARIANCE_BACKEND,
        "rank_jacobian_backend": PLUGIN_RANK_JACOBIAN_BACKEND,
        "chunk_size": int(chunk_size),
        "derivative_ridge": float(derivative_ridge),
        "device_type": str(device_type),
        "compute_dtype": str(compute_dtype),
        "intercept_mode": str(intercept_mode),
        "n_train": int(n_train),
        "predictor_dimension": int(predictor_dimension),
        "response_dimension": int(response_dimension),
        "k_references": int(k_references),
        "reference_seeds": [int(seed) for seed in reference_seeds],
        "m_ref": int(m_ref),
        "epsilon": float(epsilon),
        "fitted_sinkhorn_iters": int(fitted_sinkhorn_iters),
        "inference_code_sha256": str(inference_code_sha256),
        "fit_inventory_sha256": str(fit_inventory_sha256),
        "fit_inventory_reference_count": int(fit_inventory_reference_count),
        "fit_inventory_repetitions": [int(rep) for rep in fit_inventory_repetitions],
    }
    _validate_config_values(config)
    return config


def canonical_inference_config_json(config: Mapping[str, Any]) -> str:
    plain = {key: _plain_value(config[key]) for key in _CONFIG_KEYS}
    return json.dumps(plain, sort_keys=True, separators=(",", ":"), allow_nan=False)


def inference_config_fingerprint(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_inference_config_json(config).encode("utf-8")).hexdigest()


def inference_config_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return mirrored fields plus a canonical serialized configuration."""

    canonical = canonical_inference_config_json(config)
    metadata = {key: _plain_value(config[key]) for key in _CONFIG_KEYS}
    metadata["inference_config_json"] = canonical
    metadata["inference_config_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return metadata


def _validate_config_values(config: Mapping[str, Any]) -> None:
    missing = [key for key in _CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"inference config is missing required fields: {missing}")
    require_vectorization(config)
    if int(config["artifact_schema_version"]) != PLUGIN_COVARIANCE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported plug-in covariance artifact schema")
    if str(config["inference_method"]) != PLUGIN_COVARIANCE_METHOD:
        raise ValueError("inference config does not use plugin_covariance")
    if str(config["dual_definition"]) != PLUGIN_DUAL_DEFINITION:
        raise ValueError("inference config does not reconstruct the final-coefficient dual")
    tolerance = float(config["dual_tolerance"])
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("dual_tolerance must be finite and positive")
    if int(config["dual_max_iters"]) < 1:
        raise ValueError("dual_max_iters must be positive")
    if int(config["dual_check_every"]) < 1:
        raise ValueError("dual_check_every must be positive")
    if str(config["dual_backend_requested"]) not in {"auto", "pytorch", "triton"}:
        raise ValueError("invalid requested dual backend")
    if int(config["dual_triton_block_m"]) < 1 or int(config["chunk_size"]) < 1:
        raise ValueError("backend block and chunk sizes must be positive")
    ridge = float(config["derivative_ridge"])
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("derivative_ridge must be finite and nonnegative")
    if not _boolean(config["require_dual_convergence"], field="require_dual_convergence"):
        raise ValueError("formal plug-in covariance artifacts must require dual convergence")
    if str(config["covariance_backend"]) != PLUGIN_COVARIANCE_BACKEND:
        raise ValueError("unexpected covariance backend")
    if str(config["rank_jacobian_backend"]) != PLUGIN_RANK_JACOBIAN_BACKEND:
        raise ValueError("unexpected rank/Jacobian backend")
    if any(
        int(config[field]) < 1
        for field in ("n_train", "predictor_dimension", "response_dimension")
    ):
        raise ValueError("sample size and parameter dimensions must be positive")
    seeds = [int(seed) for seed in _plain_value(config["reference_seeds"])]
    if int(config["k_references"]) != len(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("reference seed bank does not match k_references")
    if int(config["m_ref"]) < 1 or float(config["epsilon"]) <= 0.0:
        raise ValueError("m_ref and epsilon must be positive")
    if int(config["fitted_sinkhorn_iters"]) < 1:
        raise ValueError("fitted_sinkhorn_iters must be positive")
    _require_sha256(config["inference_code_sha256"], field="inference_code_sha256")
    _require_sha256(config["fit_inventory_sha256"], field="fit_inventory_sha256")
    if int(config["fit_inventory_reference_count"]) < int(config["k_references"]):
        raise ValueError("fit inventory contains fewer members than one configured ensemble")
    repetitions = [int(rep) for rep in _plain_value(config["fit_inventory_repetitions"])]
    if not repetitions or repetitions != sorted(set(repetitions)) or any(rep < 1 for rep in repetitions):
        raise ValueError("fit_inventory_repetitions must be a sorted nonempty unique list")
    expected_count = len(repetitions) * int(config["k_references"])
    if int(config["fit_inventory_reference_count"]) != expected_count:
        raise ValueError("fit inventory reference count does not match its repetition snapshot")


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one persisted input artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file_sha256(
    metadata: Mapping[str, Any],
    path: str | Path,
    *,
    field: str,
) -> str:
    """Require a metadata digest to match the current bytes of ``path``."""

    if field not in metadata:
        raise ValueError(f"artifact metadata has no {field}")
    recorded = _require_sha256(metadata[field], field=field)
    actual = sha256_file(path)
    if recorded != actual:
        raise ValueError(f"artifact metadata field {field} does not match {path}")
    return actual


def _require_sha256(value: Any, *, field: str) -> str:
    digest = str(_plain_value(value)).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def require_plugin_dual_diagnostics(
    config: Mapping[str, Any],
    *,
    iterations: Sequence[Any],
    errors: Sequence[Any],
    backends: Sequence[Any],
    expected_count: int,
    label: str,
) -> None:
    """Validate reconstructed-dual diagnostics for final or partial artifacts."""

    iterations_plain = [int(value) for value in _plain_value(iterations)]
    errors_plain = [float(value) for value in _plain_value(errors)]
    backends_plain = [str(value) for value in _plain_value(backends)]
    if not (
        len(iterations_plain)
        == len(errors_plain)
        == len(backends_plain)
        == int(expected_count)
    ):
        raise ValueError(f"artifact {label} dual diagnostics do not have the expected count")
    if any(value < 1 or value > int(config["dual_max_iters"]) for value in iterations_plain):
        raise ValueError(f"artifact {label} dual iteration count is invalid")
    if any(
        not math.isfinite(value) or value > float(config["dual_tolerance"])
        for value in errors_plain
    ):
        raise ValueError(f"artifact {label} dual error exceeds tolerance")
    expected_backend = str(config["dual_backend_requested"])
    if expected_backend == "auto":
        expected_backend = (
            "triton"
            if str(config["device_type"]) == "cuda"
            and str(config["compute_dtype"]) == "float32"
            else "pytorch"
        )
    if any(value != expected_backend for value in backends_plain):
        raise ValueError(f"artifact {label} dual solver backend is inconsistent")


def _require_reference_provenance(metadata: Mapping[str, Any], *, k_references: int) -> None:
    for field in _REFERENCE_PROVENANCE_SHA_FIELDS:
        if field not in metadata:
            raise ValueError(f"artifact has no {field}")
        values = list(_plain_value(metadata[field]))
        if len(values) != k_references:
            raise ValueError(f"artifact {field} does not have K entries")
        for value in values:
            _require_sha256(value, field=field)
    field = "cost_scales_by_reference"
    if field not in metadata:
        raise ValueError(f"artifact has no {field}")
    scales = [float(value) for value in _plain_value(metadata[field])]
    if len(scales) != k_references:
        raise ValueError("artifact cost_scales_by_reference does not have K entries")
    if any(not math.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("artifact contains an invalid reference cost_scale")


def _require_data_provenance(metadata: Mapping[str, Any]) -> None:
    for field in _DATA_PROVENANCE_SHA_FIELDS:
        if field not in metadata:
            raise ValueError(f"artifact has no {field}")
        _require_sha256(metadata[field], field=field)


def require_plugin_covariance_config(
    metadata: Mapping[str, Any],
    *,
    expected_config: Mapping[str, Any] | None = None,
    require_final_convergence: bool = False,
) -> dict[str, Any]:
    """Validate schema, fingerprint, mirrored fields, and optional completion."""

    if "inference_config_json" not in metadata or "inference_config_sha256" not in metadata:
        raise ValueError("artifact has no complete inference configuration metadata")
    serialized = str(_plain_value(metadata["inference_config_json"]))
    recorded_fingerprint = str(_plain_value(metadata["inference_config_sha256"]))
    actual_fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if recorded_fingerprint != actual_fingerprint:
        raise ValueError("inference configuration fingerprint does not match its JSON")
    try:
        config = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid inference configuration JSON") from exc
    _validate_config_values(config)
    canonical = canonical_inference_config_json(config)
    if serialized != canonical:
        raise ValueError("inference configuration JSON is not canonical")
    for key in _CONFIG_KEYS:
        if key not in metadata:
            raise ValueError(f"artifact does not mirror inference config field {key!r}")
        expected_value = _plain_value(config[key])
        recorded = _coerce_like(metadata[key], expected_value, field=key)
        if recorded != expected_value:
            raise ValueError(f"artifact field {key!r} disagrees with its inference config")
    if expected_config is not None:
        _validate_config_values(expected_config)
        if canonical != canonical_inference_config_json(expected_config):
            raise ValueError("artifact inference configuration differs from the requested run")
    if require_final_convergence:
        for field in ("all_estimated_duals_converged", "all_oracle_duals_converged"):
            if field not in metadata or not _boolean(metadata[field], field=field):
                raise ValueError(f"artifact does not certify {field}")
        if "complete_reference_count" not in metadata:
            raise ValueError("artifact has no complete_reference_count")
        if int(_plain_value(metadata["complete_reference_count"])) != int(config["k_references"]):
            raise ValueError("artifact does not contain the configured number of references")
        k_references = int(config["k_references"])
        for prefix in ("estimated", "oracle"):
            iterations_field = f"{prefix}_dual_iterations_by_reference"
            errors_field = f"{prefix}_dual_row_log_marginal_errors"
            backends_field = f"{prefix}_dual_solver_backends_by_reference"
            for field in (iterations_field, errors_field, backends_field):
                if field not in metadata:
                    raise ValueError(f"artifact has no {field}")
            require_plugin_dual_diagnostics(
                config,
                iterations=metadata[iterations_field],
                errors=metadata[errors_field],
                backends=metadata[backends_field],
                expected_count=k_references,
                label=prefix,
            )
        _require_reference_provenance(metadata, k_references=k_references)
        _require_data_provenance(metadata)
    return config


def plugin_covariance_aggregate_metadata(
    config: Mapping[str, Any],
    *,
    noise: str,
    repetitions: int,
    repetition_npz_sha256s: Sequence[str],
) -> dict[str, Any]:
    """Return metadata for a summary artifact built from final repetition NPZs."""

    _validate_config_values(config)
    hashes = [
        _require_sha256(value, field="repetition_npz_sha256s")
        for value in repetition_npz_sha256s
    ]
    if int(repetitions) < 1 or len(hashes) != int(repetitions):
        raise ValueError("aggregate repetition count does not match its input hashes")
    canonical_inputs = json.dumps(hashes, separators=(",", ":"), allow_nan=False)
    return {
        **inference_config_metadata(config),
        "aggregate_schema_version": PLUGIN_COVARIANCE_AGGREGATE_SCHEMA_VERSION,
        "artifact_kind": PLUGIN_COVARIANCE_AGGREGATE_KIND,
        "noise": str(noise),
        "repetitions": int(repetitions),
        "repetition_npz_sha256s": hashes,
        "aggregate_inputs_sha256": hashlib.sha256(canonical_inputs.encode("utf-8")).hexdigest(),
    }


def require_plugin_covariance_aggregate(
    metadata: Mapping[str, Any],
    *,
    expected_config: Mapping[str, Any] | None = None,
    expected_noise: str | None = None,
    expected_repetitions: int | None = None,
) -> dict[str, Any]:
    """Validate the independent schema used by covariance aggregate NPZ files."""

    config = require_plugin_covariance_config(metadata, expected_config=expected_config)
    if int(_plain_value(metadata.get("aggregate_schema_version", -1))) != (
        PLUGIN_COVARIANCE_AGGREGATE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported plug-in covariance aggregate schema")
    if str(_plain_value(metadata.get("artifact_kind", ""))) != PLUGIN_COVARIANCE_AGGREGATE_KIND:
        raise ValueError("artifact is not a plug-in covariance aggregate")
    noise = str(_plain_value(metadata.get("noise", "")))
    repetitions = int(_plain_value(metadata.get("repetitions", -1)))
    hashes = [
        _require_sha256(value, field="repetition_npz_sha256s")
        for value in _plain_value(metadata.get("repetition_npz_sha256s", []))
    ]
    if repetitions < 1 or len(hashes) != repetitions:
        raise ValueError("aggregate repetition count does not match its input hashes")
    if expected_noise is not None and noise != str(expected_noise):
        raise ValueError("aggregate has the wrong noise identity")
    if expected_repetitions is not None and repetitions != int(expected_repetitions):
        raise ValueError("aggregate has the wrong repetition count")
    canonical_inputs = json.dumps(hashes, separators=(",", ":"), allow_nan=False)
    expected_inputs_hash = hashlib.sha256(canonical_inputs.encode("utf-8")).hexdigest()
    recorded_inputs_hash = _require_sha256(
        metadata.get("aggregate_inputs_sha256", ""), field="aggregate_inputs_sha256"
    )
    if recorded_inputs_hash != expected_inputs_hash:
        raise ValueError("aggregate input fingerprint is inconsistent")
    return config


__all__ = [
    "PLUGIN_COVARIANCE_ARTIFACT_SCHEMA_VERSION",
    "PLUGIN_COVARIANCE_AGGREGATE_KIND",
    "PLUGIN_COVARIANCE_AGGREGATE_SCHEMA_VERSION",
    "PLUGIN_COVARIANCE_BACKEND",
    "PLUGIN_DUAL_DEFINITION",
    "PLUGIN_RANK_JACOBIAN_BACKEND",
    "canonical_inference_config_json",
    "inference_config_fingerprint",
    "inference_config_metadata",
    "plugin_covariance_aggregate_metadata",
    "plugin_covariance_config",
    "require_plugin_covariance_aggregate",
    "require_plugin_covariance_config",
    "require_plugin_dual_diagnostics",
    "require_file_sha256",
    "sha256_file",
]
