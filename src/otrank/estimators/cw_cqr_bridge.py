"""Portable R bridge and checkpoint protocol for CW-CQR fitting."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd


CW_CQR_METHOD_ID = "cw_cqr_20levels"
_OUTPUT_FILES = (
    "beta.csv",
    "intercepts.csv",
    "fit_diagnostics.csv",
    "environment.txt",
)


def equally_spaced_quantiles(levels: int = 20) -> np.ndarray:
    """Return the formal levels ``l/(levels+1)``, ``l=1,...,levels``."""

    if int(levels) < 2:
        raise ValueError("levels must be at least two")
    return np.arange(1, int(levels) + 1, dtype=np.float64) / (int(levels) + 1)


@dataclass(frozen=True)
class CWCQRFitConfig:
    """Numerical configuration for the R ``cqrReg::cqr.fit`` bridge."""

    tau: tuple[float, ...] = tuple(equally_spaced_quantiles(20))
    method: str = "ip"
    max_iterations: int = 1000
    tolerance: float = 1e-7
    rscript: str = "Rscript"
    r_library: str | None = None

    def __post_init__(self) -> None:
        tau = np.asarray(self.tau, dtype=np.float64)
        if tau.ndim != 1 or tau.size < 2:
            raise ValueError("tau must contain at least two levels")
        if not np.all(np.diff(tau) > 0) or tau[0] <= 0 or tau[-1] >= 1:
            raise ValueError("tau must be strictly increasing inside (0, 1)")
        if self.method not in {"ip", "mm", "admm", "cd"}:
            raise ValueError(f"unsupported cqrReg method: {self.method}")
        if self.max_iterations < 1 or self.tolerance <= 0:
            raise ValueError("max_iterations and tolerance must be positive")


@dataclass(frozen=True)
class CWCQRFit:
    """Validated CW-CQR fit with coefficient matrix shape ``(d, p)``."""

    beta: np.ndarray
    quantile_intercepts: np.ndarray
    diagnostics: pd.DataFrame
    output_dir: Path
    config_hash: str
    wall_sec: float
    resumed: bool


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _bridge_resource():
    return resources.files("otrank.estimators").joinpath("fit_cw_cqr.R")


def _fit_manifest(
    x: np.ndarray,
    y: np.ndarray,
    config: CWCQRFitConfig,
) -> dict[str, object]:
    bridge_sha = hashlib.sha256(_bridge_resource().read_bytes()).hexdigest()
    r_library = config.r_library
    if r_library is None:
        # Rscript already honors R_LIBS_USER and its platform defaults. Only an
        # explicit CQR_R_LIB override needs to be forwarded to the bridge.
        r_library = os.environ.get("CQR_R_LIB") or ""
    payload: dict[str, object] = {
        "schema_version": 1,
        "method_id": CW_CQR_METHOD_ID,
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "x_sha256": _array_hash(x),
        "y_sha256": _array_hash(y),
        "tau": [float(value) for value in config.tau],
        "method": config.method,
        "max_iterations": int(config.max_iterations),
        "tolerance": float(config.tolerance),
        "rscript": config.rscript,
        "r_library": r_library,
        "bridge_sha256": bridge_sha,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["config_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _load_validated(output_dir: Path, manifest: dict[str, object], *, resumed: bool) -> CWCQRFit:
    missing = [name for name in _OUTPUT_FILES if not (output_dir / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete CW-CQR fit under {output_dir}: {missing}")
    beta = pd.read_csv(output_dir / "beta.csv").to_numpy(dtype=np.float64)
    intercepts = pd.read_csv(output_dir / "intercepts.csv").to_numpy(dtype=np.float64)
    diagnostics = pd.read_csv(output_dir / "fit_diagnostics.csv")
    d = int(manifest["y_shape"][1])  # type: ignore[index]
    p = int(manifest["x_shape"][1])  # type: ignore[index]
    levels = len(manifest["tau"])  # type: ignore[arg-type]
    if beta.shape != (d, p) or intercepts.shape != (d, levels):
        raise ValueError("CW-CQR output shapes disagree with fit manifest")
    numeric = np.concatenate(
        [beta.ravel(), intercepts.ravel(), diagnostics.select_dtypes(include=[np.number]).to_numpy().ravel()]
    )
    if not np.isfinite(numeric).all():
        raise ValueError("CW-CQR fit contains NaN or infinite values")
    wall_path = output_dir / "runtime.json"
    wall = 0.0
    if wall_path.exists():
        wall = float(json.loads(wall_path.read_text(encoding="utf-8"))["wall_sec"])
    return CWCQRFit(
        beta=beta,
        quantile_intercepts=intercepts,
        diagnostics=diagnostics,
        output_dir=output_dir,
        config_hash=str(manifest["config_hash"]),
        wall_sec=wall,
        resumed=resumed,
    )


def load_cw_cqr_fit(output_dir: Path | str) -> CWCQRFit:
    """Load and validate a fit written by :func:`fit_cw_cqr`."""

    output_dir = Path(output_dir)
    manifest_path = output_dir / "fit_config.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing CW-CQR fit_config.json under {output_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return _load_validated(output_dir, manifest, resumed=True)


def fit_cw_cqr(
    x: np.ndarray,
    y: np.ndarray,
    output_dir: Path | str,
    *,
    config: CWCQRFitConfig | None = None,
    overwrite: bool = False,
) -> CWCQRFit:
    """Fit CW-CQR through R, with input-aware and config-aware checkpointing.

    A completed fit is reused only when the hashes of ``x``, ``y``, the R
    bridge, and all numerical options match. A conflicting completed fit is
    never silently overwritten.
    """

    config = config or CWCQRFitConfig()
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("x and y must be finite two-dimensional arrays with matching rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must be finite")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _fit_manifest(x, y, config)
    manifest_path = output_dir / "fit_config.json"
    if manifest_path.exists() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != manifest["config_hash"]:
            raise ValueError(
                f"CW-CQR checkpoint under {output_dir} belongs to different data/config"
            )
        return _load_validated(output_dir, existing, resumed=True)
    if manifest_path.exists():
        # The manifest is the completion marker. Remove it before an explicit
        # overwrite so an interruption cannot expose new files under an old
        # data/config hash.
        manifest_path.unlink()

    r_library = str(manifest["r_library"])
    with tempfile.TemporaryDirectory(prefix=".cw_cqr_", dir=output_dir.parent) as temporary:
        temporary_path = Path(temporary)
        x_path = temporary_path / "x.csv"
        y_path = temporary_path / "y.csv"
        staged_output = temporary_path / "result"
        pd.DataFrame(x).to_csv(x_path, index=False)
        pd.DataFrame(y).to_csv(y_path, index=False)
        with resources.as_file(_bridge_resource()) as bridge:
            command = [
                config.rscript,
                str(bridge),
                str(x_path),
                str(y_path),
                str(staged_output),
                ",".join(f"{value:.17g}" for value in config.tau),
                config.method,
                r_library,
                str(config.max_iterations),
                f"{config.tolerance:.17g}",
            ]
            started = time.perf_counter()
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            wall = time.perf_counter() - started
        _load_validated(staged_output, manifest, resumed=False)
        for name in _OUTPUT_FILES:
            os.replace(staged_output / name, output_dir / name)

    (output_dir / "command.json").write_text(
        json.dumps(
            {
                "rscript": config.rscript,
                "bridge": "otrank.estimators/fit_cw_cqr.R",
                "method": config.method,
                "tau": list(config.tau),
                "max_iterations": config.max_iterations,
                "tolerance": config.tolerance,
                "r_library": r_library,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    (output_dir / "runtime.json").write_text(
        json.dumps({"wall_sec": wall}, indent=2), encoding="utf-8"
    )
    temporary_manifest = output_dir / ".fit_config.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return _load_validated(output_dir, manifest, resumed=False)


__all__ = [
    "CW_CQR_METHOD_ID",
    "CWCQRFit",
    "CWCQRFitConfig",
    "equally_spaced_quantiles",
    "fit_cw_cqr",
    "load_cw_cqr_fit",
]
