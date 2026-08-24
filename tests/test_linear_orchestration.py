from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_linear_dry_run_includes_cw_cqr_and_canonical_handoff(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_simulation_study.py"),
            "--config",
            str(ROOT / "configs/simulation_linear.json"),
            "--out-root",
            str(tmp_path),
            "--reps",
            "1",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "# stage=cw_cqr" in completed.stdout
    assert "otrank.experiments.run_cw_cqr_simulation" in completed.stdout
    assert str(tmp_path / "cw_cqr") in completed.stdout
    protocol = json.loads((tmp_path / "baselines/simulation_protocol.json").read_text())
    assert protocol["target"] == {
        "R": 1,
        "cw_cqr_quantiles": "tau_l=l/21, l=1,...,20",
        "d": 10,
        "n_train": 1000,
        "noises": ["A", "B", "C"],
        "p": 20,
    }
    assert protocol["coefficient_generation"]["x_center"] is False
