"""Regenerate current paper outputs, or only validate/index existing outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _outputs(root: Path) -> dict[str, tuple[str, Path]]:
    linear = root/"simulation/linear_n1000_p20_d10_final"
    relu = root/"simulation/relu_n1000_p8_d4_final"
    nhanes = root/"real_data/nhanes_p43_final"
    return {
        "linear_accuracy_table": ("linear", linear/"tables/linear_accuracy_table_rows.tex"),
        "linear_inference_table": ("linear", linear/"tables/linear_inference_table_rows.tex"),
        "linear_accuracy_figure_pdf": ("linear", linear/"figures/linear_accuracy_equivariance_boxplots.pdf"),
        "linear_accuracy_figure_png": ("linear", linear/"figures/linear_accuracy_equivariance_boxplots.png"),
        "linear_inference_figure_pdf": ("linear", linear/"figures/linear_inference_boxplots.pdf"),
        "linear_inference_figure_png": ("linear", linear/"figures/linear_inference_boxplots.png"),
        "relu_prediction_table": ("relu", relu/"tables/relu_prediction_table_rows.tex"),
        "nhanes_table": ("nhanes", nhanes/"tables/nhanes_table.tex"),
        "nhanes_predictors_data": ("nhanes", nhanes/"tables/nhanes_predictors.csv"),
        "nhanes_pairwise_stability_table": ("nhanes", nhanes/"tables/nhanes_pairwise_stability.tex"),
        "nhanes_semisynthetic_table": ("nhanes", nhanes/"supplement/semisynthetic_invariance/comparison/table_s2_4.tex"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, help="Generated-results root; defaults to <project-root>/results.")
    parser.add_argument("--only", choices=["linear", "relu", "nhanes"])
    parser.add_argument("--skip-build", action="store_true", help="Only validate/index existing outputs; do not regenerate figures or tables.")
    args = parser.parse_args()
    project = args.project_root.resolve()
    root = (args.output_root or project/"results").resolve()
    outputs = {name: path for name, (study, path) in _outputs(root).items() if args.only is None or study == args.only}
    if not args.skip_build:
        from . import current_paper_outputs as current
        destinations = {"linear": root/"simulation/linear_n1000_p20_d10_final",
                        "relu": root/"simulation/relu_n1000_p8_d4_final",
                        "nhanes": root/"real_data/nhanes_p43_final"}
        for study in ([args.only] if args.only else destinations):
            getattr(current, study)(project, destinations[study])
    missing = [str(path) for path in outputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing current paper outputs:\n"+"\n".join(missing))
    root.mkdir(parents=True, exist_ok=True)
    index = {name: str(path.relative_to(root)) for name, path in outputs.items()}
    (root/"paper_output_index.json").write_text(json.dumps(index, indent=2)+"\n")
    (root/"PAPER_OUTPUT_INDEX.md").write_text(
        "# Current paper outputs\n\nStable semantic keys do not depend on manuscript numbering.\n\n"
        + "\n".join(f"- `{name}`: `{path}`" for name, path in index.items())+"\n")
    verb = "Validated and indexed existing" if args.skip_build else "Regenerated from frozen metrics and indexed"
    print(f"{verb} {len(outputs)} current paper outputs. No estimators were refitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
