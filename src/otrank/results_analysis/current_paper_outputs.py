"""Generate current paper outputs from compact, versioned, non-participant CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .paper_data import load_frame, linear_point, nhanes_equivariance, validate_grid


def _directories(root):
    for folder in ("metrics", "summaries", "tables", "figures"):
        (root/folder).mkdir(parents=True, exist_ok=True)


def linear(project: Path, root: Path) -> None:
    from .finalize_linear_simulation import INFERENCE_METHODS, summarize, write_table3_rows, write_table4_rows
    from .plot_linear_accuracy_equivariance import build_figure
    from .plot_linear_inference_95_99 import build_figure as inference_figure

    _directories(root)
    point = linear_point(project)
    inference = load_frame("linear_inference.csv", project)
    validate_grid(inference, INFERENCE_METHODS)
    if not inference.loc[inference.method == "OTRR", "coverage_95"].between(0, 1).all():
        raise ValueError("invalid centered inference coverage")
    point_summary = summarize(point, ["coef_error", "e_max"])
    inference_summary = summarize(inference, ["avg_interval_length_95", "coverage_95", "avg_interval_length_99", "coverage_99"])
    point.to_csv(root/"metrics/point_estimation_by_rep.csv", index=False)
    inference.to_csv(root/"metrics/inference_by_rep.csv", index=False)
    point_summary.to_csv(root/"summaries/point_estimation_summary.csv", index=False)
    inference_summary.to_csv(root/"summaries/inference_summary.csv", index=False)
    write_table3_rows(point_summary, root/"tables/linear_accuracy_table_rows.tex")
    write_table4_rows(inference_summary, root/"tables/linear_inference_table_rows.tex")
    for stem, figure in [("linear_accuracy_equivariance_boxplots", build_figure(point)),
                         ("linear_inference_boxplots", inference_figure(inference))]:
        figure.savefig(root/f"figures/{stem}.pdf")
        figure.savefig(root/f"figures/{stem}.png", dpi=350)
        plt.close(figure)


def relu(project: Path, root: Path) -> None:
    from .finalize_relu_simulation import METHODS, summarize, write_table_rows

    _directories(root)
    frame = load_frame("relu_prediction.csv", project)
    validate_grid(frame, METHODS)
    summary = summarize(frame)
    frame.to_csv(root/"metrics/prediction_by_rep.csv", index=False)
    summary.to_csv(root/"summaries/prediction_summary.csv", index=False)
    write_table_rows(summary, root/"tables/relu_prediction_table_rows.tex")


def nhanes(project: Path, root: Path) -> None:
    from .nhanes_p43_final import _write_table5_tex, _write_pairwise_stability_tex
    from .compare_nhanes_semisynthetic_otrr_fixedref import write_latex_table

    _directories(root)
    summary = load_frame("nhanes_summary.csv", project).merge(nhanes_equivariance(project), on="method", validate="one_to_one")
    summary.to_csv(root/"tables/nhanes_table.csv", index=False)
    _write_table5_tex(root/"tables/nhanes_table.tex", summary)
    load_frame("nhanes_equivariance.csv", project).to_csv(root/"metrics/equivariance_by_fold_rotation.csv", index=False)
    pairwise = load_frame("nhanes_pairwise.csv", project)
    pairwise.to_csv(root/"summaries/pairwise_stability.csv", index=False)
    _write_pairwise_stability_tex(root/"tables/nhanes_pairwise_stability.tex", pairwise)
    load_frame("nhanes_predictors.csv", project).to_csv(root/"tables/nhanes_predictors.csv", index=False)
    repetitions = load_frame("semisynthetic_repetitions.csv", project)
    ratio = load_frame("semisynthetic_ratio.csv", project)
    keys = ["noise", "kappa", "method", "data_provenance"]
    if repetitions.duplicated(keys+["repetition"]).any() or len(repetitions) != 120:
        raise ValueError("semi-synthetic data require 6 methods x 2 signal levels x 10 repetitions")
    summary = repetitions.groupby(keys, as_index=False).agg(
        repetitions=("repetition", "size"), tpr_mean=("tpr", "mean"), tpr_sd=("tpr", "std"),
        fdr_mean=("fdr", "mean"), fdr_sd=("fdr", "std"), iou_mean=("iou", "mean"), iou_sd=("iou", "std"),
        intersection_mean=("intersection", "mean"), union_mean=("union", "mean"))
    if not summary.repetitions.eq(10).all():
        raise ValueError("incomplete semi-synthetic repetition groups")
    summary = summary.merge(ratio, on=keys, validate="one_to_one")
    np.testing.assert_allclose(summary.intersection_mean, summary.average_intersection, rtol=1e-12)
    np.testing.assert_allclose(summary.union_mean, summary.average_union, rtol=1e-12)
    np.testing.assert_allclose(summary.ratio_of_averages, summary.average_intersection/summary.average_union, rtol=1e-12)
    destination = root/"supplement/semisynthetic_invariance/comparison"
    destination.mkdir(parents=True, exist_ok=True)
    repetitions.to_csv(destination/"repetition_metrics.csv", index=False)
    summary.to_csv(destination/"summary.csv", index=False)
    write_latex_table(summary.sort_values(["kappa", "method"]), destination)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["linear", "relu", "nhanes"], required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.study](args.project_root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
