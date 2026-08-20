"""Sequential, dependency-aware, and resource-aware simulation scheduling."""

from __future__ import annotations

import csv
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing as mp
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
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

import torch
from tqdm import tqdm

from otrank.data.noise import make_generator, random_orthogonal
from otrank.data.sim import (
    gaussian_nonlinear_jacobian,
    make_linear_simulation,
    make_nonlinear_simulation,
)
from otrank.estimators import fit_coordinatewise_huber, fit_radial_huber, fit_zero_intercept_ols
from otrank.metrics import (
    coefficient_error,
    centered_prediction_mse,
    frobenius_angle,
    jacobian_relative_gap,
    jacobian_rmse_per_entry,
    jacobian_rowspace_diagnostics,
    mean_bias,
    prediction_mae,
    prediction_mse,
    prediction_mse_per_dim,
    prediction_rmse_per_dim,
    prediction_space_angle,
    principal_angles,
    rotation_prediction_gap,
    rotation_prediction_mse_per_dim,
    rotation_prediction_rmse_per_dim,
    translation_adjusted_mse,
)

_PROGRESS_FILE = None

from .core import (
    _dtype_from_name, _load_ot_rank_iteration_fit,
    _make_simulation_from_config, _run_one,
)
from .layout import (
    _atomic_torch_save, _atomic_write_json,
    _candidate_init_labels, _candidate_init_out_dirs,
    _canonical_method_label, _ensemble_metrics_path, _ensemble_state_path,
    _init_fit_path_for_label, _load_checkpoint, _load_completed_job_rows,
    _method_label, _noise_label, _principal_angle_stats, _refresh_partial_summary,
    _rep_dir, _require_existing_init, _rowspace_principal_angle_stats,
    _safe_component,
)
from .monitoring import _progress_file
from .ensemble import (
    _resource_clone_ot_method, _resource_completed_rows,
    _resource_ensemble_row, _resource_eval_iters,
)


def _artifact_root(config: dict[str, Any], metrics_path: Path) -> Path:
    value = config.get("_artifact_root")
    return Path(str(value)) if value is not None else metrics_path.parent

def _make_jobs(config: dict[str, Any]) -> list[tuple[int, int, int, dict[str, Any], dict[str, Any]]]:
    seeds = [int(s) for s in config.get("seeds", [1])]
    rep_indices = [int(value) for value in config.get("rep_indices", list(range(1, len(seeds) + 1)))]
    if len(rep_indices) != len(seeds):
        raise ValueError("config rep_indices must have the same length as seeds")
    noises = list(config["noises"])
    methods = list(config["methods"])
    return [
        (job_index, rep_index, seed, noise, method)
        for job_index, (rep_index, seed, noise, method) in enumerate(
            (rep_index, seed, noise, method)
            for rep_index, seed in zip(rep_indices, seeds, strict=True)
            for noise in noises
            for method in methods
        )
    ]


def _worker(
    worker_id: int,
    job_queue,
    config: dict[str, Any],
    out_dir: Path,
    result_queue,
) -> None:
    bar = tqdm(
        total=1,
        desc=f"worker {worker_id}",
        position=worker_id + 1,
        leave=False,
        dynamic_ncols=True,
        file=_progress_file(),
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}",
    )
    try:
        while True:
            try:
                job_index, rep_index, seed, noise, method = job_queue.get_nowait()
            except queue.Empty:
                break
            noise_label = _noise_label(noise)
            method_label = _method_label(method)
            task_label = f"{noise_label}/{method_label}/rep{rep_index:03d}"
            bar.reset(total=1)
            bar.set_description_str(f"worker {worker_id} {task_label}", refresh=False)
            bar.set_postfix_str("starting", refresh=True)
            bar.refresh()

            try:
                rows = _run_one(
                    config,
                    rep_index,
                    seed,
                    noise,
                    method,
                    job_index=job_index,
                    out_dir=out_dir,
                    progress_callback=None,
                )
                result_queue.put({"type": "result", "rows": rows})
            except Exception as exc:  # pragma: no cover - reported to parent process
                result_queue.put(
                    {
                        "type": "error",
                        "worker_id": worker_id,
                        "seed": seed,
                        "rep_index": rep_index,
                        "noise": noise_label,
                        "method": method_label,
                        "error": repr(exc),
                    }
                )
            finally:
                if int(bar.total or 0) != 1:
                    bar.reset(total=1)
                else:
                    bar.n = 0
                bar.set_description_str(f"worker {worker_id} {task_label}", refresh=False)
                bar.set_postfix_str("done", refresh=False)
                bar.update(1)
                bar.refresh()
    finally:
        bar.set_description_str(f"worker {worker_id} idle", refresh=False)
        bar.set_postfix_str("", refresh=True)
        bar.close()
        result_queue.put({"type": "done", "worker_id": worker_id})


def _run_parallel(
    config: dict[str, Any],
    jobs: list[tuple[int, int, int, dict[str, Any], dict[str, Any]]],
    metrics_path: Path,
    workers: int,
    append: bool = False,
) -> None:
    start_method = str(config.get("runtime", {}).get("multiprocessing_start_method", "fork"))
    try:
        ctx = mp.get_context(start_method)
    except ValueError:
        ctx = mp.get_context("spawn")
    job_queue: mp.Queue = ctx.Queue()
    for job in jobs:
        job_queue.put(job)
    result_queue: mp.Queue = ctx.Queue()
    processes = [
        ctx.Process(target=_worker, args=(worker_id, job_queue, config, _artifact_root(config, metrics_path), result_queue), daemon=False)
        for worker_id in range(min(workers, len(jobs)))
    ]
    for process in processes:
        process.start()

    done = 0
    errors: list[dict[str, Any]] = []
    with metrics_path.open("a" if append else "w") as fh, tqdm(
        total=len(jobs),
        desc="total",
        position=0,
        dynamic_ncols=True,
        file=_progress_file(),
    ) as total_bar:
        while done < len(processes):
            msg = result_queue.get()
            if msg["type"] == "result":
                for row in msg["rows"]:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
                total_bar.update(1)
            elif msg["type"] == "error":
                errors.append(msg)
                total_bar.update(1)
            elif msg["type"] == "done":
                done += 1

    for process in processes:
        process.join()
    if errors:
        (metrics_path.parent / "errors.json").write_text(json.dumps(errors, indent=2, sort_keys=True))
        raise RuntimeError(f"{len(errors)} simulation jobs failed; see {metrics_path.parent / 'errors.json'}")


def _run_threaded(
    config: dict[str, Any],
    jobs: list[tuple[int, int, int, dict[str, Any], dict[str, Any]]],
    metrics_path: Path,
    workers: int,
    append: bool = False,
) -> None:
    job_queue: queue.Queue = queue.Queue()
    for job in jobs:
        job_queue.put(job)
    result_queue: queue.Queue = queue.Queue()
    threads = [
        threading.Thread(
            target=_worker,
            args=(worker_id, job_queue, config, _artifact_root(config, metrics_path), result_queue),
            daemon=False,
        )
        for worker_id in range(min(workers, len(jobs)))
    ]
    for thread in threads:
        thread.start()

    done = 0
    errors: list[dict[str, Any]] = []
    with metrics_path.open("a" if append else "w") as fh, tqdm(
        total=len(jobs),
        desc="total",
        position=0,
        dynamic_ncols=True,
        file=_progress_file(),
    ) as total_bar:
        while done < len(threads):
            msg = result_queue.get()
            if msg["type"] == "result":
                for row in msg["rows"]:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
                total_bar.update(1)
            elif msg["type"] == "error":
                errors.append(msg)
                total_bar.update(1)
            elif msg["type"] == "done":
                done += 1

    for thread in threads:
        thread.join()
    if errors:
        (metrics_path.parent / "errors.json").write_text(json.dumps(errors, indent=2, sort_keys=True))
        raise RuntimeError(f"{len(errors)} simulation jobs failed; see {metrics_path.parent / 'errors.json'}")


def _run_sequential(
    config: dict[str, Any],
    jobs: list[tuple[int, int, int, dict[str, Any], dict[str, Any]]],
    metrics_path: Path,
    append: bool = False,
) -> None:
    with metrics_path.open("a" if append else "w") as fh:
        for job_index, rep_index, seed, noise, method in tqdm(
            jobs,
            desc="total",
            position=0,
            dynamic_ncols=True,
            file=_progress_file(),
        ):
            rows = _run_one(
                config,
                rep_index,
                seed,
                noise,
                method,
                job_index=job_index,
                out_dir=_artifact_root(config, metrics_path),
            )
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()


def _regular_output_key(
    data_cfg: dict[str, Any],
    noise: dict[str, Any],
    rep_index: int,
    label: str,
) -> tuple[str, str, int, str]:
    return (str(data_cfg.get("n_train")), _noise_label(noise), int(rep_index), str(label))


def _regular_task_dependencies(
    config: dict[str, Any],
    job: tuple[int, int, int, dict[str, Any], dict[str, Any]],
) -> list[str]:
    _, _, _, _, method = job
    method_name = str(method.get("name"))
    params = dict(method.get("params", {}))
    deps: list[str] = []
    if method_name.startswith("ot_rank") or method_name == "relu_pnw_fullrisk":
        deps.extend(_candidate_init_labels(params))
    return deps


def _regular_task_ready(
    config: dict[str, Any],
    out_dir: Path,
    job: tuple[int, int, int, dict[str, Any], dict[str, Any]],
    failed_regular: set[tuple[str, str, int, str]],
) -> tuple[bool, str | None]:
    _, rep_index, _, noise, method = job
    data_cfg = config["data"]
    deps = _regular_task_dependencies(config, job)
    if not deps:
        return True, None
    method_name = str(method.get("name"))
    params = dict(method.get("params", {}))
    if method_name.startswith("ot_rank") or method_name == "relu_pnw_fullrisk":
        for label in deps:
            for search_dir in _candidate_init_out_dirs(out_dir, params):
                if _init_fit_path_for_label(search_dir, data_cfg, noise, rep_index, label).exists():
                    return True, None
        if _require_existing_init(params):
            return False, f"required existing init fit not found; tried labels {deps}"
        if all(_regular_output_key(data_cfg, noise, rep_index, label) in failed_regular for label in deps):
            return False, f"required init fit not available; tried labels {deps}"
        return False, None
    for label in deps:
        key = _regular_output_key(data_cfg, noise, rep_index, label)
        if key in failed_regular:
            return False, f"dependency {label} failed"
        if not _init_fit_path_for_label(out_dir, data_cfg, noise, rep_index, label).exists():
            return False, None
    return True, None


def _dynamic_task_priority(task: dict[str, Any]) -> int:
    method = task["job"][4]
    method_name = str(method.get("name"))
    if method_name in {"coordinatewise_asm", "relu_coordinatewise_asm"}:
        return 0
    if method_name.startswith("ot_rank") or method_name == "relu_pnw_fullrisk":
        return 20
    if method_name == "mcqr_ot_lens":
        return 40
    if method_name in {
        "zero_intercept_ols",
        "coordinatewise_huber",
        "radial_huber",
        "relu_least_squares",
        "relu_coordinatewise_huber",
        "relu_radial_huber",
    }:
        return 50
    return 45


def _dependency_worker_process(
    worker_id: int,
    task_queue,
    config: dict[str, Any],
    out_dir: Path,
    result_queue,
) -> None:
    while True:
        task = task_queue.get()
        if isinstance(task, dict) and task.get("kind") == "__stop__":
            break
        job_index, rep_index, seed, noise, method = task["job"]
        try:
            rows = _run_one(
                config,
                rep_index,
                seed,
                noise,
                method,
                job_index=job_index,
                out_dir=out_dir,
                progress_callback=None,
            )
            result_queue.put({"type": "result", "task": task, "rows": rows})
        except Exception as exc:
            result_queue.put(
                {
                    "type": "error",
                    "task": task,
                    "worker_id": worker_id,
                    "seed": seed,
                    "rep_index": rep_index,
                    "noise": _noise_label(noise),
                    "method": _method_label(method),
                    "error": repr(exc),
                }
            )
    result_queue.put({"type": "done", "worker_id": worker_id})


def _run_dependency_threaded(
    config: dict[str, Any],
    jobs: list[tuple[int, int, int, dict[str, Any], dict[str, Any]]],
    metrics_path: Path,
    workers: int,
    append: bool = False,
    use_processes: bool = False,
) -> None:
    data_cfg = config["data"]
    runtime_cfg = config.get("runtime", {})
    tasks = [{"kind": "regular", "job": job} for job in jobs]

    if use_processes:
        start_method = str(runtime_cfg.get("dependency_start_method", "spawn"))
        try:
            ctx = mp.get_context(start_method)
        except ValueError:
            ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()
    else:
        ctx = None
        task_queue = queue.Queue()
        result_queue = queue.Queue()
    stop_task = {"kind": "__stop__"}

    def task_label(task: dict[str, Any]) -> str:
        _, rep_index, _, noise, method = task["job"]
        return "/".join([_noise_label(noise), _method_label(method), f"rep{rep_index:03d}"])

    def task_ready(
        task: dict[str, Any],
        failed_regular: set[tuple[str, str, int, str]],
    ) -> tuple[bool, str | None]:
        return _regular_task_ready(config, _artifact_root(config, metrics_path), task["job"], failed_regular)

    def dynamic_worker(worker_id: int) -> None:
        disable_worker_progress = bool(runtime_cfg.get("disable_worker_progress", use_processes))
        bar = tqdm(
            total=1,
            desc=f"worker {worker_id}",
            position=worker_id + 1,
            leave=False,
            dynamic_ncols=True,
            file=_progress_file(),
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}",
            disable=disable_worker_progress,
        )
        try:
            while True:
                task = task_queue.get()
                if isinstance(task, dict) and task.get("kind") == "__stop__":
                    break
                job_index, rep_index, seed, noise, method = task["job"]
                label = task_label(task)
                bar.reset(total=1)
                bar.set_description_str(f"worker {worker_id} {label}", refresh=False)
                bar.set_postfix_str("starting", refresh=True)

                try:
                    rows = _run_one(
                        config,
                        rep_index,
                        seed,
                        noise,
                        method,
                        job_index=job_index,
                        out_dir=_artifact_root(config, metrics_path),
                        progress_callback=None,
                    )
                    result_queue.put({"type": "result", "task": task, "rows": rows})
                except Exception as exc:
                    result_queue.put(
                        {
                            "type": "error",
                            "task": task,
                            "worker_id": worker_id,
                            "seed": seed,
                            "rep_index": rep_index,
                            "noise": _noise_label(noise),
                            "method": _method_label(method),
                            "error": repr(exc),
                        }
                    )
                finally:
                    if int(bar.total or 0) != 1:
                        bar.reset(total=1)
                    else:
                        bar.n = 0
                    bar.set_description_str(f"worker {worker_id} {label}", refresh=False)
                    bar.set_postfix_str("done", refresh=False)
                    bar.update(1)
                    bar.refresh()
        finally:
            bar.set_description_str(f"worker {worker_id} idle", refresh=False)
            bar.set_postfix_str("", refresh=True)
            bar.close()
            result_queue.put({"type": "done", "worker_id": worker_id})

    pending = tasks[:]
    in_flight = 0
    completed = 0
    errors: list[dict[str, Any]] = []
    failed_regular: set[tuple[str, str, int, str]] = set()
    n_workers = min(workers, len(tasks))
    if use_processes:
        workers_started = [
            ctx.Process(  # type: ignore[union-attr]
                target=_dependency_worker_process,
                args=(worker_id, task_queue, config, _artifact_root(config, metrics_path), result_queue),
                daemon=False,
            )
            for worker_id in range(n_workers)
        ]
    else:
        workers_started = [
            threading.Thread(target=dynamic_worker, args=(worker_id,), daemon=False)
            for worker_id in range(n_workers)
        ]
    for worker_obj in workers_started:
        worker_obj.start()

    def dispatch_ready(total_bar: tqdm) -> None:
        nonlocal in_flight, pending, completed
        while in_flight < n_workers and pending:
            blocked: list[tuple[dict[str, Any], str]] = []
            ready: list[dict[str, Any]] = []
            waiting: list[dict[str, Any]] = []
            for task in pending:
                ok, reason = task_ready(task, failed_regular)
                if ok:
                    ready.append(task)
                elif reason is not None:
                    blocked.append((task, reason))
                else:
                    waiting.append(task)
            for task, reason in blocked:
                _, rep_index, seed, noise, method = task["job"]
                errors.append(
                    {
                        "type": "error",
                        "seed": seed,
                        "rep_index": rep_index,
                        "noise": _noise_label(noise),
                        "method": _method_label(method),
                        "error": reason,
                    }
                )
                failed_regular.add(_regular_output_key(data_cfg, noise, rep_index, _method_label(method)))
            if blocked:
                pending = waiting + ready
                completed += len(blocked)
                total_bar.update(len(blocked))
                continue
            if not ready:
                pending = waiting
                break
            ready.sort(key=_dynamic_task_priority)
            task = ready[0]
            pending = [item for item in waiting + ready[1:] if item is not task]
            task_queue.put(task)
            in_flight += 1

    with metrics_path.open("a" if append else "w") as fh, tqdm(
        total=len(tasks),
        desc="total",
        position=0,
        dynamic_ncols=True,
        file=_progress_file(),
    ) as total_bar:
        dispatch_ready(total_bar)
        while completed < len(tasks):
            if in_flight == 0 and pending:
                errors.append({"type": "error", "error": "No ready tasks remain; unresolved dependency cycle or missing init."})
                total_bar.update(len(pending))
                completed += len(pending)
                pending = []
                break
            msg = result_queue.get()
            if msg["type"] == "done":
                continue
            in_flight -= 1
            task = msg["task"]
            if msg["type"] == "result":
                for row in msg.get("rows", []):
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
            elif msg["type"] == "error":
                errors.append({key: value for key, value in msg.items() if key != "task"})
                _, rep_index, _, noise, method = task["job"]
                failed_regular.add(_regular_output_key(data_cfg, noise, rep_index, _method_label(method)))
            completed += 1
            total_bar.update(1)
            dispatch_ready(total_bar)

    for _ in workers_started:
        task_queue.put(stop_task)
    for worker_obj in workers_started:
        worker_obj.join()
    if errors:
        (metrics_path.parent / "errors.json").write_text(json.dumps(errors, indent=2, sort_keys=True))
        raise RuntimeError(f"{len(errors)} simulation tasks failed; see {metrics_path.parent / 'errors.json'}")


def _resource_set_runtime(config: dict[str, Any], *, device: str, dtype: str, torch_threads: int) -> dict[str, Any]:
    local = dict(config)
    runtime = dict(local.get("runtime", {}))
    runtime.update(
        {
            "device": str(device),
            "dtype": str(dtype),
            "torch_num_threads": int(torch_threads),
            "generate_on_cpu": True,
        }
    )
    local["runtime"] = runtime
    return local


def _resource_stop_file(out_dir: Path | str, task_id: str) -> Path:
    return Path(out_dir) / "task_stops" / f"{task_id}.stop"


def _resource_run_task(
    *,
    config: dict[str, Any],
    out_dir: str,
    rep_index: int,
    seed: int,
    noise: dict[str, Any],
    method: dict[str, Any],
    device: str,
    dtype: str,
    torch_threads: int,
    job_index: int,
    task_id: str,
) -> dict[str, Any]:
    stop_path = _resource_stop_file(out_dir, task_id)
    try:
        stop_path.unlink()
    except FileNotFoundError:
        pass
    old_stop_file = os.environ.get("OTRANK_STOP_FILE")
    os.environ["OTRANK_STOP_FILE"] = str(stop_path)
    try:
        local = _resource_set_runtime(config, device=device, dtype=dtype, torch_threads=torch_threads)
        local["methods"] = [method]
        rows = _run_one(
            local,
            int(rep_index),
            int(seed),
            noise,
            method,
            job_index=int(job_index),
            out_dir=Path(out_dir),
            progress_callback=None,
        )
        return {"task_id": task_id, "status": "done", "rows": rows}
    except InterruptedError as exc:
        return {"task_id": task_id, "status": "preempted", "error": repr(exc), "rows": []}
    except Exception as exc:  # noqa: BLE001
        return {"task_id": task_id, "status": "failed", "error": repr(exc), "rows": []}
    finally:
        if old_stop_file is None:
            os.environ.pop("OTRANK_STOP_FILE", None)
        else:
            os.environ["OTRANK_STOP_FILE"] = old_stop_file


def _resource_run_batch_task(
    *,
    config: dict[str, Any],
    out_dir: str,
    rep_index: int,
    seed: int,
    noise: dict[str, Any],
    methods: list[dict[str, Any]],
    device: str,
    dtype: str,
    torch_threads: int,
    job_index: int,
    task_id: str,
    reference_seeds: list[int],
) -> dict[str, Any]:
    """Run multiple reference-seed OT fits in one long-lived worker process."""

    all_rows: list[dict[str, Any]] = []
    completed_reference_seeds: list[int] = []
    for offset, (reference_seed, method) in enumerate(zip(reference_seeds, methods, strict=True)):
        child_task_id = f"{task_id}_ref{int(reference_seed)}"
        payload = _resource_run_task(
            config=config,
            out_dir=out_dir,
            rep_index=int(rep_index),
            seed=int(seed),
            noise=noise,
            method=method,
            device=device,
            dtype=dtype,
            torch_threads=int(torch_threads),
            job_index=int(job_index) * 1000 + int(offset),
            task_id=child_task_id,
        )
        rows = list(payload.get("rows") or [])
        all_rows.extend(rows)
        if payload.get("status") == "done":
            completed_reference_seeds.append(int(reference_seed))
            continue
        return {
            "task_id": task_id,
            "status": payload.get("status", "failed"),
            "error": payload.get("error", f"reference seed {reference_seed} failed"),
            "rows": all_rows,
            "reference_seeds": list(reference_seeds),
            "completed_reference_seeds": completed_reference_seeds,
        }
    return {
        "task_id": task_id,
        "status": "done",
        "rows": all_rows,
        "reference_seeds": list(reference_seeds),
        "completed_reference_seeds": completed_reference_seeds,
    }


def _resource_append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_fields: list[str] = []
    existing_rows: list[dict[str, str]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = list(reader)
    fields = list(existing_fields)
    seen = set(fields)
    for key in row:
        if key not in seen:
            fields.append(key)
            seen.add(key)
    mode = "w" if not existing_fields or fields != existing_fields else "a"
    with path.open(mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
            if existing_rows:
                writer.writerows(existing_rows)
        writer.writerow(row)


def _resource_append_rows(metrics_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


















def _run_resource_scheduled(
    config: dict[str, Any],
    jobs: list[tuple[int, int, int, dict[str, Any], dict[str, Any]]],
    metrics_path: Path,
    workers: int,
) -> None:
    runtime_cfg = dict(config.get("runtime", {}))
    dtype = str(runtime_cfg.get("dtype", "float32"))
    cpu_workers = int(runtime_cfg.get("resource_cpu_workers", runtime_cfg.get("cpu_workers", workers)))
    gpu_workers = int(runtime_cfg.get("resource_gpu_workers", 1))
    gpu_torch_threads = int(runtime_cfg.get("resource_gpu_torch_threads", runtime_cfg.get("gpu_torch_threads", 1)))
    gpu_batch_reference_seeds = bool(runtime_cfg.get("gpu_batch_reference_seeds", True))
    reference_flush_every = max(1, int(runtime_cfg.get("reference_flush_every", 5)))
    reference_ensemble_k = int(runtime_cfg.get("reference_ensemble_k", 1))
    reference_seed_start = int(runtime_cfg.get("reference_seed_start", 20260721))
    report_ensemble_k = [int(value) for value in runtime_cfg.get("report_ensemble_k", [reference_ensemble_k])]
    report_ensemble_k = sorted({k for k in report_ensemble_k if 1 <= k <= max(1, reference_ensemble_k)})
    if reference_flush_every > 0:
        report_ensemble_k = sorted(
            set(report_ensemble_k)
            | set(range(reference_flush_every, max(1, reference_ensemble_k) + 1, reference_flush_every))
        )
    if not report_ensemble_k:
        report_ensemble_k = [max(1, reference_ensemble_k)]
    reference_seeds = [reference_seed_start + offset for offset in range(max(1, reference_ensemble_k))]
    out_dir = _artifact_root(config, metrics_path)
    data_cfg = config["data"]

    ot_jobs = [job for job in jobs if str(job[4].get("name", "")).startswith("ot_rank")]
    regular_jobs = [job for job in jobs if job not in ot_jobs]
    if not regular_jobs and not ot_jobs:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.touch(exist_ok=True)
        return
    init_labels = {label for job in ot_jobs for label in _candidate_init_labels(dict(job[4].get("params", {})))}

    completed_regular: set[tuple[str, int, str]] = set()
    completed_ot_seeds: set[tuple[str, int, str, int]] = set()
    completed_ensembles: set[tuple[str, int, str, int, int]] = set()
    for _, rep_index, _, noise, method in regular_jobs:
        if _resource_completed_rows(out_dir, data_cfg, noise, rep_index, method) is not None:
            completed_regular.add((_noise_label(noise), int(rep_index), _method_label(method)))
    for _, rep_index, _, noise, method in ot_jobs:
        base_label = _method_label(method)
        eval_iters = _resource_eval_iters(method)
        for reference_index, reference_seed in enumerate(reference_seeds, 1):
            seed_method = _resource_clone_ot_method(
                method,
                reference_seed=reference_seed,
                reference_index=reference_index,
            )
            if _resource_completed_rows(out_dir, data_cfg, noise, rep_index, seed_method) is not None:
                completed_ot_seeds.add((_noise_label(noise), int(rep_index), base_label, int(reference_seed)))
        # Ensemble rows are cheap and can be recomputed if absent.
        _ = eval_iters
    if metrics_path.exists():
        with metrics_path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(row.get("method_family", "")) != "ot_rank_pnw_reference_ensemble":
                    continue
                try:
                    completed_ensembles.add(
                        (
                            str(row["noise"]),
                            int(row["rep_index"]),
                            str(row.get("method_folder", row.get("method", "")))
                            .split("_refensemble", 1)[0]
                            .split("_K", 1)[0],
                            int(row["ensemble_k"]),
                            int(row["iteration"]),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue

    cpu_pending = sorted(
        [job for job in regular_jobs if (_noise_label(job[3]), int(job[1]), _method_label(job[4])) not in completed_regular],
        key=lambda job: (0 if _method_label(job[4]) in init_labels else 1, int(job[1]), _noise_label(job[3]), _method_label(job[4])),
    )
    errors: list[dict[str, Any]] = []
    cpu_futures: dict[Any, dict[str, Any]] = {}
    gpu_futures: dict[Any, dict[str, Any]] = {}
    running_ot: set[tuple[str, int, str, int]] = set()
    job_counter = 0

    def submit(executor, fmap, *, job, method, device: str, threads: int, reference_seed: int | None = None):
        nonlocal job_counter
        job_index, rep_index, seed, noise, _ = job
        job_counter += 1
        task_label = _method_label(method)
        seed_suffix = f"_ref{reference_seed}" if reference_seed is not None else ""
        task_id = "_".join(
            [
                _safe_component(_noise_label(noise)),
                f"rep{int(rep_index):03d}",
                _safe_component(task_label),
                str(device),
                str(seed_suffix).strip("_"),
            ]
        )
        fut = executor.submit(
            _resource_run_task,
            config=config,
            out_dir=str(out_dir),
            rep_index=int(rep_index),
            seed=int(seed),
            noise=noise,
            method=method,
            device=device,
            dtype=dtype,
            torch_threads=int(threads),
            job_index=job_counter if reference_seed is None else job_counter * 1000 + int(reference_seed),
            task_id=task_id,
        )
        fmap[fut] = {
            "job": job,
            "method": method,
            "device": device,
            "threads": int(threads),
            "reference_seed": reference_seed,
            "task_id": task_id,
            "job_index": job_counter,
        }

    def append_event(spec: dict[str, Any], payload: dict[str, Any]) -> None:
        job = spec["job"]
        _, rep_index, seed, noise, _ = job
        method = spec["method"]
        reference_seed_value = spec.get("reference_seed")
        if reference_seed_value is None and spec.get("reference_seeds"):
            reference_seed_value = ",".join(str(value) for value in spec.get("reference_seeds", []))
        _resource_append_csv(
            out_dir / "scheduler_events.csv",
            {
                "timestamp": time.time(),
                "task_id": spec.get("task_id"),
                "status": payload.get("status"),
                "error": payload.get("error"),
                "noise": _noise_label(noise),
                "rep_index": int(rep_index),
                "seed": int(seed),
                "method_label": _method_label(method),
                "reference_seed": reference_seed_value,
                "device": spec.get("device"),
                "threads": spec.get("threads"),
                "job_index": spec.get("job_index"),
            },
        )

    def collect(fmap: dict[Any, dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        if not fmap:
            return []
        done, _ = wait(list(fmap), timeout=0.25, return_when=FIRST_COMPLETED)
        out = []
        for fut in done:
            spec = fmap.pop(fut)
            payload = fut.result()
            append_event(spec, payload)
            if payload.get("status") == "failed":
                errors.append({"spec": {k: str(v) for k, v in spec.items() if k != "job"}, "payload": payload})
            _resource_append_rows(metrics_path, list(payload.get("rows") or []))
            out.append((spec, payload))
        return out

    def all_regular_done() -> bool:
        expected = {(_noise_label(noise), int(rep_index), _method_label(method)) for _, rep_index, _, noise, method in regular_jobs}
        return expected.issubset(completed_regular)

    def ot_ready(job) -> bool:
        _, rep_index, _, noise, method = job
        labels = _candidate_init_labels(dict(method.get("params", {})))
        return any((_noise_label(noise), int(rep_index), label) in completed_regular for label in labels)

    def ot_missing_required_init(job) -> str | None:
        _, rep_index, _, noise, method = job
        params = dict(method.get("params", {}))
        if not _require_existing_init(params):
            return None
        labels = _candidate_init_labels(params)
        if any((_noise_label(noise), int(rep_index), label) in completed_regular for label in labels):
            return None
        for label in labels:
            for search_dir in _candidate_init_out_dirs(out_dir, params):
                if _init_fit_path_for_label(search_dir, data_cfg, noise, rep_index, label).exists():
                    completed_regular.add((_noise_label(noise), int(rep_index), label))
                    return None
        return f"required existing init fit not found for noise={_noise_label(noise)} rep={int(rep_index)} labels={labels}"

    def all_ot_done() -> bool:
        for job in ot_jobs:
            _, rep_index, _, noise, method = job
            base_label = _method_label(method)
            for k in report_ensemble_k:
                for iteration in _resource_eval_iters(method):
                    if (_noise_label(noise), int(rep_index), base_label, int(k), int(iteration)) not in completed_ensembles:
                        return False
        return True

    def maybe_write_ensemble_rows(job) -> None:
        _, rep_index, seed, noise, method = job
        base_label = _method_label(method)
        done_seeds = {
            reference_seed
            for reference_seed in reference_seeds
            if (_noise_label(noise), int(rep_index), base_label, int(reference_seed)) in completed_ot_seeds
        }
        rows: list[dict[str, Any]] = []
        for iteration in _resource_eval_iters(method):
            for k in report_ensemble_k:
                key = (_noise_label(noise), int(rep_index), base_label, int(k), int(iteration))
                if key in completed_ensembles:
                    continue
                required = set(reference_seeds[: int(k)])
                if not required.issubset(done_seeds):
                    continue
                rows.append(
                    _resource_ensemble_row(
                        config=config,
                        out_dir=out_dir,
                        rep_index=int(rep_index),
                        seed=int(seed),
                        noise=noise,
                        base_method=method,
                        reference_seeds=reference_seeds,
                        k=int(k),
                        iteration=int(iteration),
                        runtime_sec=float("nan"),
                        dtype_name=dtype,
                    )
                )
                completed_ensembles.add(key)
        _resource_append_rows(metrics_path, rows)

    for job in ot_jobs:
        maybe_write_ensemble_rows(job)

    def submit_cpu_ready(cpu_pool) -> None:
        while cpu_pending and len(cpu_futures) < max(1, cpu_workers):
            job = cpu_pending.pop(0)
            submit(cpu_pool, cpu_futures, job=job, method=job[4], device="cpu", threads=1)

    def next_ot_seed_task() -> tuple[tuple[int, int, int, dict[str, Any], dict[str, Any]], dict[str, Any], int] | None:
        for job in ot_jobs:
            missing = ot_missing_required_init(job)
            if missing is not None:
                raise FileNotFoundError(missing)
            _, rep_index, _, noise, method = job
            if not ot_ready(job):
                continue
            base_label = _method_label(method)
            for reference_index, reference_seed in enumerate(reference_seeds, 1):
                key = (_noise_label(noise), int(rep_index), base_label, int(reference_seed))
                if key in completed_ot_seeds or key in running_ot:
                    continue
                return (
                    job,
                    _resource_clone_ot_method(
                        method,
                        reference_seed=reference_seed,
                        reference_index=reference_index,
                    ),
                    int(reference_seed),
                )
        return None

    def next_ot_batch_task() -> tuple[tuple[int, int, int, dict[str, Any], dict[str, Any]], list[dict[str, Any]], list[int]] | None:
        for job in ot_jobs:
            missing = ot_missing_required_init(job)
            if missing is not None:
                raise FileNotFoundError(missing)
            _, rep_index, _, noise, method = job
            if not ot_ready(job):
                continue
            base_label = _method_label(method)
            seeds: list[int] = []
            methods: list[dict[str, Any]] = []
            for reference_index, reference_seed in enumerate(reference_seeds, 1):
                key = (_noise_label(noise), int(rep_index), base_label, int(reference_seed))
                if key in completed_ot_seeds or key in running_ot:
                    continue
                seeds.append(int(reference_seed))
                methods.append(
                    _resource_clone_ot_method(
                        method,
                        reference_seed=reference_seed,
                        reference_index=reference_index,
                    )
                )
                if len(seeds) >= reference_flush_every:
                    break
            if seeds:
                return job, methods, seeds
        return None

    def submit_gpu_if_idle(gpu_pool) -> None:
        if len(gpu_futures) >= max(1, gpu_workers):
            return
        if gpu_batch_reference_seeds:
            task = next_ot_batch_task()
            if task is None:
                return
            nonlocal job_counter
            job, methods, batch_reference_seeds = task
            _, rep_index, seed, noise, base_method = job
            base_label = _method_label(base_method)
            for reference_seed in batch_reference_seeds:
                running_ot.add((_noise_label(noise), int(rep_index), base_label, int(reference_seed)))
            job_counter += 1
            task_id = "_".join(
                [
                    _safe_component(_noise_label(noise)),
                    f"rep{int(rep_index):03d}",
                    _safe_component(base_label),
                    "cuda",
                    f"refbatch{len(batch_reference_seeds)}",
                ]
            )
            fut = gpu_pool.submit(
                _resource_run_batch_task,
                config=config,
                out_dir=str(out_dir),
                rep_index=int(rep_index),
                seed=int(seed),
                noise=noise,
                methods=methods,
                device="cuda",
                dtype=dtype,
                torch_threads=gpu_torch_threads,
                job_index=job_counter,
                task_id=task_id,
                reference_seeds=batch_reference_seeds,
            )
            gpu_futures[fut] = {
                "job": job,
                "method": base_method,
                "device": "cuda",
                "threads": int(gpu_torch_threads),
                "reference_seed": None,
                "reference_seeds": list(batch_reference_seeds),
                "task_id": task_id,
                "job_index": job_counter,
                "batch": True,
            }
            return
        task = next_ot_seed_task()
        if task is None:
            return
        job, method, reference_seed = task
        _, rep_index, _, noise, base_method = job
        running_ot.add((_noise_label(noise), int(rep_index), _method_label(base_method), int(reference_seed)))
        submit(gpu_pool, gpu_futures, job=job, method=method, device="cuda", threads=gpu_torch_threads, reference_seed=reference_seed)

    total_expected = len(regular_jobs) + len(ot_jobs) * max(1, reference_ensemble_k)
    with ProcessPoolExecutor(max_workers=max(1, cpu_workers)) as cpu_pool, ProcessPoolExecutor(max_workers=max(1, gpu_workers)) as gpu_pool, tqdm(
        total=total_expected,
        desc="resource-scheduled",
        dynamic_ncols=True,
        file=_progress_file(),
    ) as bar:
        last_count = len(completed_regular) + len(completed_ot_seeds)
        submit_cpu_ready(cpu_pool)
        while cpu_futures or gpu_futures or cpu_pending or not all_ot_done():
            for spec, payload in collect(cpu_futures):
                _, rep_index, _, noise, method = spec["job"]
                if payload.get("status") == "done":
                    completed_regular.add((_noise_label(noise), int(rep_index), _method_label(method)))
                submit_cpu_ready(cpu_pool)
            for spec, payload in collect(gpu_futures):
                _, rep_index, _, noise, base_method = spec["job"]
                base_label = _method_label(base_method)
                if spec.get("batch"):
                    for reference_seed in spec.get("reference_seeds", []):
                        key = (_noise_label(noise), int(rep_index), base_label, int(reference_seed))
                        running_ot.discard(key)
                    if payload.get("status") == "done":
                        done_seeds = payload.get("completed_reference_seeds") or spec.get("reference_seeds", [])
                        for reference_seed in done_seeds:
                            completed_ot_seeds.add((_noise_label(noise), int(rep_index), base_label, int(reference_seed)))
                        maybe_write_ensemble_rows(spec["job"])
                else:
                    key = (_noise_label(noise), int(rep_index), base_label, int(spec["reference_seed"]))
                    running_ot.discard(key)
                    if payload.get("status") == "done":
                        completed_ot_seeds.add(key)
                        maybe_write_ensemble_rows(spec["job"])
            submit_gpu_if_idle(gpu_pool)
            current_count = len(completed_regular) + len(completed_ot_seeds)
            if current_count > last_count:
                bar.update(current_count - last_count)
                last_count = current_count
                _refresh_partial_summary(out_dir)
            if errors:
                (out_dir / "scheduler_errors.json").write_text(json.dumps(errors, indent=2, sort_keys=True))
                raise RuntimeError(f"{len(errors)} resource-scheduled tasks failed; see {out_dir / 'scheduler_errors.json'}")
            time.sleep(0.1)
    _refresh_partial_summary(out_dir)
