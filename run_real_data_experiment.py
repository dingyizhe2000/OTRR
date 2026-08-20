"""Run or finalize a configured paper real-data experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "main_real_data.json"


def _run(args: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> None:
    print(" ".join(args))
    if not dry_run:
        subprocess.run(args, check=True, env=env)


def _local_package_env() -> dict[str, str]:
    """Make the src-layout package importable without requiring installation."""
    env = os.environ.copy()
    source_root = str(ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source_root if not existing else os.pathsep.join((source_root, existing))
    return env


def _run_nhanes_final(config: dict, args: argparse.Namespace) -> None:
    expected = {
        "sample_size": 8237,
        "predictor_dimension": 43,
        "response_dimension": 7,
        "composition_dimension": 8,
        "zero_replacement": "train_half_q01",
        "outer_folds": 5,
        "split_seed": 20260726,
        "vectorization": "column-major vec(B)",
        "survey_weights": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"NHANES config must use {key}={value!r}")
    otrr = config.get("otrr", {})
    expected_otrr = {
        "initializer": "CW-CQR",
        "m_ref": 65536,
        "K": 10,
        "epsilon": 0.1,
        "steps": 500,
        "learning_rate": 0.0001,
        "inference": "plugin_covariance",
        "inference_residuals": "estimated",
        "dual_tolerance": 0.001,
        "dual_max_iterations": 500,
        "dual_check_every": 5,
    }
    for key, value in expected_otrr.items():
        if otrr.get(key) != value:
            raise ValueError(f"NHANES OTRR config must use {key}={value!r}")
    result_root = args.result_root
    if result_root is None:
        result_root = (ROOT / str(config["result_root"])).resolve()
    bootstrap = config.get("bootstrap", {})
    command = [
        sys.executable,
        "-m",
        "otrank.results_analysis.nhanes_p43_final",
        "--root",
        str(result_root),
        "--bootstrap-repetitions",
        str(int(bootstrap.get("repetitions", 2000))),
        "--bootstrap-seed",
        str(int(bootstrap.get("seed", 20260727))),
    ]
    _run(command, dry_run=args.dry_run, env=_local_package_env())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-root", type=Path, default=ROOT / "results")
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    experiment = config.get("experiment")
    if experiment == "nhanes_p43_oral_microbiome_d7_final":
        _run_nhanes_final(config, args)
    else:
        raise ValueError(f"unsupported real-data experiment: {experiment!r}")


if __name__ == "__main__":
    main()
