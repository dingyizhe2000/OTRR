"""Regenerate every empirical table and figure from the three final bundles."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from otrank.project_layout import code_root


def _run(command: list[str], *, project_root: Path) -> None:
    env = os.environ.copy()
    source = str(code_root(project_root) / "src")
    env["PYTHONPATH"] = source if not env.get("PYTHONPATH") else os.pathsep.join((source, env["PYTHONPATH"]))
    subprocess.run(command, cwd=project_root, env=env, check=True)


def _outputs(project_root: Path) -> dict[str, str]:
    linear = project_root / "results/simulation/linear_n1000_p20_d10_final"
    relu = project_root / "results/simulation/relu_n1000_p8_d4_final"
    nhanes = project_root / "results/real_data/nhanes_p43_final"
    semi = nhanes / "supplement/semisynthetic_invariance/comparison"
    return {
        "table_3": str(linear / "tables/table3_rows.tex"),
        "table_4": str(linear / "tables/table4_rows.tex"),
        "table_5": str(nhanes / "tables/table5.tex"),
        "figure_s2_1_pdf": str(linear / "figures/linear_accuracy_invariance_boxplots.pdf"),
        "figure_s2_1_png": str(linear / "figures/linear_accuracy_invariance_boxplots.png"),
        "figure_s2_2_pdf": str(linear / "figures/linear_inference_boxplots.pdf"),
        "figure_s2_2_png": str(linear / "figures/linear_inference_boxplots.png"),
        "table_s2_1": str(relu / "tables/relu_prediction_table_rows.tex"),
        "table_s2_2_data": str(nhanes / "tables/table_s2_2_predictors.csv"),
        "table_s2_2_status": str(nhanes / "tables/table_s2_2_STATUS.md"),
        "table_s2_3": str(nhanes / "tables/table_s2_3.tex"),
        "table_s2_4": str(semi / "table_s2_4.tex"),
    }


def _write_index(project_root: Path, outputs: dict[str, str]) -> None:
    result_root = project_root / "results"
    relative = {key: str(Path(value).relative_to(project_root)) for key, value in outputs.items()}
    (result_root / "paper_output_index.json").write_text(json.dumps(relative, indent=2) + "\n")
    lines = [
        "# Paper output index",
        "",
        "All empirical paper outputs are regenerated from the three canonical result bundles.",
        "Current manuscript Table 2 is theoretical and is intentionally outside this empirical pipeline.",
        "",
        "| Output | Generated file |",
        "|---|---|",
    ]
    lines.extend(f"| `{key}` | `{value}` |" for key, value in relative.items())
    lines.extend(
        [
            "",
            "Table S2.2 currently has a machine-readable predictor list and an explicit status file; ",
            "its final grouped LaTeX presentation remains intentionally unresolved.",
            "",
        ]
    )
    (result_root / "PAPER_OUTPUT_INDEX.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--skip-build", action="store_true", help="Only validate and index existing outputs.")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    source_root = code_root(project_root)

    if not args.skip_build:
        python = sys.executable
        _run(
            [python, "-m", "otrank.results_analysis.finalize_linear_simulation", "--project-root", str(project_root), "--output-dir", str(project_root / "results/simulation/linear_n1000_p20_d10_final")],
            project_root=project_root,
        )
        _run(
            [python, "-m", "otrank.results_analysis.finalize_relu_simulation", "--project-root", str(project_root), "--output-dir", str(project_root / "results/simulation/relu_n1000_p8_d4_final")],
            project_root=project_root,
        )
        _run(
            [python, str(source_root / "run_real_data_experiment.py"), "--config", str(source_root / "configs/main_real_data.json"), "--result-root", str(project_root / "results/real_data/nhanes_p43_final")],
            project_root=project_root,
        )
        nhanes = project_root / "results/real_data/nhanes_p43_final"
        _run(
            [
                python,
                "-m",
                "otrank.results_analysis.compare_nhanes_semisynthetic_otrr_fixedref",
                "--baseline-root",
                str(nhanes / "supplement/semisynthetic_invariance/baselines"),
                "--otrr-root",
                str(nhanes / "supplement/semisynthetic_invariance/otrr_noiseC_fixedref"),
                "--output",
                str(nhanes / "supplement/semisynthetic_invariance/comparison"),
            ],
            project_root=project_root,
        )

    outputs = _outputs(project_root)
    missing = [path for path in outputs.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("missing paper outputs:\n" + "\n".join(missing))
    _write_index(project_root, outputs)
    print(f"Regenerated and indexed {len(outputs)} paper outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
