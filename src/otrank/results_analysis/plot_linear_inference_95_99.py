"""Plot paper-facing 95%/99% interval lengths and coverage from final CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["LS", "Radial Huber", "CW-Huber", "CW-ASM", "CW-CQR", "OTRR"]
COLORS = {
    "LS": "#88CCEE",
    "Radial Huber": "#117733",
    "CW-Huber": "#44AA99",
    "CW-ASM": "#CCBB44",
    "CW-CQR": "#AA4499",
    "OTRR": "#332288",
}
METRICS = [
    "avg_interval_length_95",
    "coverage_95",
    "avg_interval_length_99",
    "coverage_99",
]


def _panel(ax, frame: pd.DataFrame, metric: str, rng: np.random.Generator) -> None:
    positions = np.arange(len(METHODS), dtype=float)
    for position, method in zip(positions, METHODS):
        values = frame.loc[frame["method"] == method, metric].to_numpy(float)
        if values.size != 50:
            raise ValueError(f"{method} has {values.size} values for {metric}, expected 50")
        ax.boxplot(
            [values],
            positions=[position],
            widths=0.58,
            whis=1.5,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.25},
            boxprops={
                "facecolor": COLORS[method],
                "edgecolor": "#333333",
                "linewidth": 0.8,
            },
            whiskerprops={"color": "#555555", "linewidth": 0.8},
            capprops={"color": "#555555", "linewidth": 0.8},
        )
        jitter = rng.uniform(-0.16, 0.16, size=values.size)
        ax.scatter(
            np.full(values.size, position) + jitter,
            values,
            s=10,
            color=COLORS[method],
            edgecolors="none",
            alpha=0.38,
            zorder=3,
        )
    ax.set_xticks(positions, METHODS, rotation=30, ha="right", rotation_mode="anchor")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8.0)


def build_figure(data: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(4, 3, figsize=(16.0, 14.0), constrained_layout=True)
    rng = np.random.default_rng(20260811)
    labels = iter([f"({chr(ord('a') + index)})" for index in range(12)])
    length_values = data[["avg_interval_length_95", "avg_interval_length_99"]].to_numpy(float)
    length_limits = (float(length_values.min()) / 1.12, float(length_values.max()) * 1.12)
    row_labels = {
        "avg_interval_length_95": "Average 95% CI length",
        "coverage_95": "95% coefficientwise coverage",
        "avg_interval_length_99": "Average 99% CI length",
        "coverage_99": "99% coefficientwise coverage",
    }

    for column, noise in enumerate("ABC"):
        block = data[data["noise"] == noise]
        for row, metric in enumerate(METRICS):
            ax = axes[row, column]
            _panel(ax, block, metric, rng)
            ax.text(
                0.02,
                0.025,
                next(labels),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=9.5,
            )
            if metric.startswith("avg_interval_length"):
                ax.set_yscale("log")
                ax.set_ylim(*length_limits)
            else:
                ax.set_ylim(0.875 if metric == "coverage_99" else 0.80, 1.005)
                nominal = 0.95 if metric == "coverage_95" else 0.99
                ax.axhline(
                    nominal,
                    color="#555555",
                    linestyle="--",
                    linewidth=1.0,
                    zorder=1,
                )
            if column == 0:
                suffix = " (log scale)" if metric.startswith("avg_interval_length") else ""
                ax.set_ylabel(row_labels[metric] + suffix, fontsize=9.5)
        axes[0, column].set_title(f"Noise {noise}", fontsize=11.5, pad=8)
    return figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(
            "results/simulation/linear_n1000_p20_d10_final/"
            "metrics/inference_by_rep.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/simulation/linear_n1000_p20_d10_final/figures"
        ),
    )
    args = parser.parse_args()
    project = args.project_root.resolve()
    input_csv = args.input_csv if args.input_csv.is_absolute() else project / args.input_csv
    output = args.output_dir if args.output_dir.is_absolute() else project / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_csv)
    required = {"noise", "rep", "method", *METRICS}
    if not required.issubset(data.columns):
        raise ValueError(f"final inference CSV is missing {sorted(required - set(data.columns))}")
    observed = data.groupby(["noise", "method"])["rep"].nunique().to_dict()
    expected = {(noise, method): 50 for noise in "ABC" for method in METHODS}
    if observed != expected or data.duplicated(["noise", "method", "rep"]).any():
        raise ValueError("final inference CSV is not a complete 3 x 6 x 50 grid")
    if not np.isfinite(data[METRICS].to_numpy(float)).all():
        raise ValueError("final inference CSV contains a non-finite plotting metric")

    figure = build_figure(data)
    figure.savefig(output / "linear_inference_boxplots.pdf", bbox_inches="tight")
    figure.savefig(output / "linear_inference_boxplots.png", dpi=350, bbox_inches="tight")
    plt.close(figure)

    audit = {
        "input_csv": str(input_csv),
        "metrics": METRICS,
        "interval_length_scale": "log",
        "repetitions_per_noise_method": 50,
        "rows": int(len(data)),
    }
    (output / "plot_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
