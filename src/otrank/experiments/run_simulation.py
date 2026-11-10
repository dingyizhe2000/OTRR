"""Run the paper's linear OT-rank simulation experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_NUMERIC_THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "MALLOC_ARENA_MAX": "2",
}
for _env_name, _env_value in _NUMERIC_THREAD_ENV_DEFAULTS.items():
    os.environ.setdefault(_env_name, _env_value)

_PAPER_NOISE_LABELS = {"A", "B", "C"}
_PAPER_METHOD_NAMES = {
    "linear_free_intercept",
    "zero_intercept_ols",
    "coordinatewise_huber",
    "radial_huber",
    "coordinatewise_asm",
    "mcqr_ot_lens",
    "ot_rank_pnw_fullrisk",
    "relu_pnw_fullrisk",
    "relu_least_squares",
    "relu_coordinatewise_huber",
    "relu_radial_huber",
    "relu_coordinatewise_asm",
    "relu_coordinatewise_cqr",
}
_PAPER_METHOD_LABELS = {
    "ls_free_intercept",
    "coordinatewise_huber_free_intercept",
    "radial_huber_free_intercept",
    "zero_intercept_ols",
    "coordinatewise_huber_tau1p345",
    "radial_huber_tau1p345_sqrtd",
    "coordinatewise_asm_zero_intercept",
    "mcqr_ot_lens",
    "mcqr_gaussian",
    "mcqr_hypercube",
    "ot_rank_ls_init",
    "ot_rank_cw_huber_init",
    "ot_rank_radial_huber_init",
    "ot_rank_cw_asm_init",
    "ot_rank_mcqr_init",
    "relu_pnw_fullrisk",
    "relu_least_squares",
    "relu_coordinatewise_huber",
    "relu_radial_huber",
    "relu_coordinatewise_asm_huber_init",
    "relu_coordinatewise_cqr",
}

from otrank.experiments.layout import _method_label, _noise_label, _refresh_partial_summary
from otrank.experiments.monitoring import _SimulationMonitor
from otrank.experiments.scheduling import (
    _make_jobs, _run_dependency_threaded, _run_parallel,
    _run_resource_scheduled, _run_sequential, _run_threaded,
)


def _run_metadata_dir(out_dir: Path, config: dict[str, Any]) -> Path:
    data = config["data"]
    noises = "-".join(sorted(_noise_label(noise) for noise in config.get("noises", [])))
    rep_indices = [int(value) for value in config.get("rep_indices", [])]
    rep_part = f"rep{min(rep_indices):03d}_{max(rep_indices):03d}" if rep_indices else "repall"
    dgp = str(data.get("dgp", "linear"))
    identity = json.dumps(
        {key: value for key, value in config.items() if not str(key).startswith("_")},
        sort_keys=True,
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    name = f"{dgp}_n{data['n_train']}_p{data['p']}_d{data['d']}_noise{noises}_{rep_part}_{digest}"
    return out_dir / ".runs" / name

def _validate_paper_config(config: dict[str, Any]) -> None:
    data = config.get("data", {})
    dims = (int(data.get("p", -1)), int(data.get("d", -1)))
    allow_exploratory = bool(config.get("allow_exploratory", False))
    dgp = str(data.get("dgp", "linear")).lower()
    allowed_dims = {(8, 4)} if dgp == "gaussian_nonlinear" else {(20, 10), (30, 20)}
    if not allow_exploratory and dims not in allowed_dims:
        raise ValueError(f"Public simulation runner does not support dimensions {dims} for DGP {dgp!r}")
    if not allow_exploratory and int(data.get("n_train", -1)) not in {500, 1000}:
        raise ValueError("Public simulation runner only supports paper sample sizes n=500 and n=1000")
    if dgp not in {"linear", "gaussian_nonlinear"}:
        raise ValueError("Public simulation runner only supports linear and Gaussian nonlinear DGPs")

    noise_labels = {_noise_label(noise) for noise in config.get("noises", [])}
    if not allow_exploratory and dgp == "linear" and (not noise_labels or not noise_labels.issubset(_PAPER_NOISE_LABELS)):
        raise ValueError(f"Public simulation runner only supports paper noise labels A/B/C; got {sorted(noise_labels)}")
    if not allow_exploratory and dgp == "gaussian_nonlinear" and (not noise_labels or not noise_labels.issubset(_PAPER_NOISE_LABELS)):
        raise ValueError(f"Gaussian nonlinear simulations support noise labels A/B/C; got {sorted(noise_labels)}")
    for method in config.get("methods", []):
        name = str(method.get("name", ""))
        label = _method_label(method)
        if name not in _PAPER_METHOD_NAMES:
            raise ValueError(f"Public simulation runner does not support method name {name!r}")
        if not allow_exploratory and label not in _PAPER_METHOD_LABELS:
            raise ValueError(f"Public simulation runner does not support method label {label!r}")


def run(config: dict[str, Any], out_dir: Path, workers: int = 5) -> None:
    config = dict(config)
    _validate_paper_config(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    config["_artifact_root"] = str(out_dir.resolve())
    metadata_dir = _run_metadata_dir(out_dir, config)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    config_to_save = {key: value for key, value in config.items() if not str(key).startswith("_")}
    (metadata_dir / "config.json").write_text(json.dumps(config_to_save, indent=2, sort_keys=True))
    metrics_path = metadata_dir / "metrics.jsonl"
    if (not metrics_path.exists() or metrics_path.stat().st_size == 0) and (metadata_dir / "errors.json").exists():
        (metadata_dir / "errors.json").unlink()
    _refresh_partial_summary(out_dir)
    jobs = _make_jobs(config)
    with _SimulationMonitor(metadata_dir, dict(config.get("runtime", {}))):
        if bool(config.get("runtime", {}).get("resource_scheduler", False)):
            _run_resource_scheduled(config, jobs, metrics_path, workers)
            return
        init_method_names = {
            "zero_intercept_ols",
            "coordinatewise_huber",
            "radial_huber",
            "coordinatewise_asm",
        }
        init_jobs = [job for job in jobs if job[4]["name"] in init_method_names]
        remaining_jobs = [job for job in jobs if job[4]["name"] not in init_method_names]
        if workers <= 1:
            _run_sequential(config, init_jobs + remaining_jobs, metrics_path)
        else:
            backend = str(config.get("runtime", {}).get("parallel_backend", "thread")).lower()
            runner = _run_threaded if backend == "thread" else _run_parallel if backend == "process" else None
            if runner is None:
                raise ValueError("runtime.parallel_backend must be 'thread' or 'process'")
            if backend == "thread" and bool(config.get("runtime", {}).get("dependency_scheduler", True)):
                dependency_backend = str(config.get("runtime", {}).get("dependency_backend", "thread")).lower()
                if dependency_backend not in {"thread", "process"}:
                    raise ValueError("runtime.dependency_backend must be 'thread' or 'process'")
                _run_dependency_threaded(
                    config,
                    jobs,
                    metrics_path,
                    workers,
                    append=False,
                    use_processes=(dependency_backend == "process"),
                )
                return
            phases = [phase for phase in (init_jobs, remaining_jobs) if phase]
            for phase_index, phase in enumerate(phases):
                phase_workers = max(1, min(len(phase), workers))
                runner(config, phase, metrics_path, phase_workers, append=phase_index > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    workers = int(args.workers if args.workers is not None else config.get("runtime", {}).get("workers", 5))
    run(config, args.out, workers=max(1, workers))


if __name__ == "__main__":
    main()
