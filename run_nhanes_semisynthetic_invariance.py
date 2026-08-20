#!/usr/bin/env python3
"""Run the resumable NHANES d=7 semi-synthetic invariance experiment."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd


CODE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = CODE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from otrank.project_layout import project_root  # noqa: E402

PROJECT_ROOT = project_root(CODE_ROOT)

from otrank.experiments.nhanes_semisynthetic_invariance import (  # noqa: E402
    ExperimentConfig,
    config_payload,
    dgp_seed,
    iter_tasks,
    prepare_template,
    run_task,
    summarize,
)


DEFAULT_OUTPUT = PROJECT_ROOT / (
    "results/real_data/nhanes_p43_final/supplement/semisynthetic_invariance/baselines"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start-repetition", type=int, default=1)
    parser.add_argument("--end-repetition", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--single-task", nargs=3, metavar=("REP", "NOISE", "KAPPA"))
    return parser.parse_args()


def environment_audit(config: ExperimentConfig) -> dict[str, object]:
    r_check = subprocess.run(
        [
            "Rscript",
            "-e",
            (
                f'.libPaths(c("{config.cqr_r_library}",.libPaths())); '
                'cat(as.character(packageVersion("asm"))," ",'
                'as.character(packageVersion("cqrReg")),"\\n")'
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    import numpy as np
    import scipy
    import torch

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "r_packages": r_check.stdout.strip().splitlines()[-1],
        "workers": config.workers,
        "blas_threads": 1,
    }


def write_interim(output: Path, config_sha256: str, completed_tasks: int) -> None:
    metrics, sets = summarize(output, config_sha256)
    payload = {
        "completed_tasks": completed_tasks,
        "complete_repetitions": int(
            metrics.groupby("repetition").size().eq(1 * 2 * 3 * 5).sum()
        ) if not metrics.empty else 0,
        "rotation_metric_rows": len(metrics),
        "set_metric_rows": len(sets),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    target = output / "summaries/interim_status.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def run_phase(
    output: Path,
    config: ExperimentConfig,
    config_sha256: str,
    repetitions: range,
    *,
    interim_every_tasks: int = 75,
) -> int:
    tasks = list(iter_tasks(repetitions))
    completed = 0
    context = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=config.workers,
        mp_context=context,
    ) as pool:
        futures = {
            pool.submit(run_task, PROJECT_ROOT, output, task, config, config_sha256): task
            for task in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(f"task failed: {task.task_id}")
            completed += 1
            print(
                f"complete {completed}/{len(tasks)} {result['task_id']} "
                f"wall={result['wall_sec']:.1f}s",
                flush=True,
            )
            if completed % interim_every_tasks == 0:
                write_interim(output, config_sha256, completed)
    write_interim(output, config_sha256, completed)
    return completed


def main() -> None:
    args = parse_args()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = replace(
        ExperimentConfig(),
        repetitions=int(args.repetitions),
        workers=int(args.workers),
    )
    provenance_path = output / "prepared/template_provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    else:
        provenance = prepare_template(PROJECT_ROOT, output)
    payload = config_payload(config, provenance)
    config_path = output / "execution_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != payload["config_sha256"]:
            raise ValueError("output root belongs to a different experiment configuration")
    else:
        config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        (output / "environment.json").write_text(
            json.dumps(environment_audit(config), indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "exact_commands.txt").write_text(
            shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
            + "\n\nResume/summarize:\n"
            + "PYTHONPATH=" + shlex.quote(str(SOURCE_ROOT)) + " "
            + shlex.join(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output",
                    str(output),
                    "--repetitions",
                    str(config.repetitions),
                    "--workers",
                    str(config.workers),
                ]
            )
            + "\n"
            + "PYTHONPATH=" + shlex.quote(str(SOURCE_ROOT)) + " "
            + shlex.join(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output",
                    str(output),
                    "--repetitions",
                    str(config.repetitions),
                    "--workers",
                    str(config.workers),
                    "--summarize-only",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        inventory = pd.DataFrame(
            [
                {
                    "task_id": task.task_id,
                    "repetition": task.repetition,
                    "noise": task.noise,
                    "kappa": task.kappa,
                    "dgp_seed": dgp_seed(config, task),
                    "expected_representations": 3,
                    "expected_methods": 5,
                    "expected_fit_artifacts": 15,
                }
                for task in iter_tasks(range(1, config.repetitions + 1))
            ]
        )
        inventory.to_csv(output / "task_inventory.csv", index=False)
        (output / "REPORT.md").write_text(
            "# NHANES d=7 semi-synthetic invariance experiment\n\n"
            "Status: initialized; formal R=10 execution is resumable and incomplete until "
            "`COMPLETE.marker` exists.\n\n"
            "The fixed design has n=8237 and p=43. The d=7 signal template is the mean "
            "of 15 mapped-back formal CW-CQR-initialized OTRR K=10 fold/representation "
            "ensembles, sparsified to the 71 genus-vs-Other contrasts that satisfy the "
            "paper's strict five-fold BH q<0.05 rule under all three formal response "
            "representations. This aggregate is used because no separate full-sample d=7 "
            "OTRR fit is available.\n\n"
            "Each method is independently refit under the identity and two fixed Haar "
            "rotations. Coefficients and analytic covariance matrices are mapped back using "
            "Q.T and I_p kron Q.T under column-major vec(B).\n",
            encoding="utf-8",
        )
    config_sha256 = str(payload["config_sha256"])
    if args.prepare_only:
        print(json.dumps(provenance, indent=2))
        return
    if args.summarize_only:
        metrics, sets = summarize(output, config_sha256)
        print(f"summarized rotation_rows={len(metrics)} set_rows={len(sets)}")
        return
    if args.single_task:
        from otrank.experiments.nhanes_semisynthetic_invariance import TaskSpec

        rep, noise, kappa = args.single_task
        print(run_task(PROJECT_ROOT, output, TaskSpec(int(rep), noise, float(kappa)), config, config_sha256))
        summarize(output, config_sha256)
        return

    end = int(args.end_repetition or args.repetitions)
    start = int(args.start_repetition)
    if start < 1 or end > args.repetitions or start > end:
        raise ValueError("invalid repetition range")

    run_phase(output, config, config_sha256, range(start, end + 1))
    summarize(output, config_sha256)
    (output / "COMPLETE.marker").write_text(
        f"repetitions={end - start + 1} config_sha256={config_sha256}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
