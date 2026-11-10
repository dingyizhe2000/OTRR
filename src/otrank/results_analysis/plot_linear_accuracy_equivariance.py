"""Current coefficient-error/E_max boxplots from actual repetition-level values."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

from .paper_data import linear_point

METHODS = ["LS", "Radial Huber", "CW-Huber", "CW-ASM", "CW-CQR", "MCQR", "OTRR"]
COLORS = {"LS": "#88CCEE", "Radial Huber": "#117733", "CW-Huber": "#44AA99",
          "CW-ASM": "#CCBB44", "CW-CQR": "#AA4499", "MCQR": "#999933", "OTRR": "#332288"}


@plt.rc_context()
def build_figure(data):
    if not np.isfinite(data[["coef_error", "e_max"]]).all().all() or (data[["coef_error", "e_max"]] <= 0).any().any():
        raise ValueError("Log-scale plots require finite positive values; do not silently floor or omit observations.")
    plt.rcdefaults()
    # Draw at the manuscript's final width rather than scaling down a wide canvas.
    figure, axes = plt.subplots(2, 3, figsize=(6.5, 3.65), sharey="row")
    figure.subplots_adjust(left=0.133, right=0.989, top=0.90, bottom=0.13, wspace=0.10, hspace=0.27)
    rng = np.random.default_rng(20260811)
    for column, noise in enumerate("ABC"):
        block = data[data.noise == noise]
        for row, (metric, label) in enumerate([("coef_error", "Coefficient error"), ("e_max", "Coefficient equivariance error")]):
            ax = axes[row, column]
            for position, method in enumerate(METHODS):
                values = block.loc[block.method == method, metric].to_numpy(float)
                if len(values) != 50:
                    raise ValueError(f"expected 50 actual observations for {noise}/{method}/{metric}")
                ax.boxplot([values], positions=[position], widths=0.58, whis=1.5, patch_artist=True, showfliers=False,
                    medianprops={"color": "#111111", "linewidth": 1.2},
                    boxprops={"facecolor": COLORS[method], "edgecolor": "#333333", "linewidth": 0.8},
                    whiskerprops={"color": "#555555", "linewidth": 0.8}, capprops={"color": "#555555", "linewidth": 0.8})
                ax.scatter(np.full(len(values), position)+rng.uniform(-0.16, 0.16, len(values)), values,
                           s=2.4, color=COLORS[method], edgecolors="none", alpha=0.38, zorder=3)
            ax.set_yscale("log")
            ax.set_xticks(range(len(METHODS)), [])
            ax.tick_params(axis="both", length=2.5, width=0.55, labelsize=8.5, pad=2)
            ax.tick_params(axis="y", labelleft=column == 0)
            ax.yaxis.set_minor_locator(NullLocator())
            if column == 0:
                ax.set_ylabel(label if row == 0 else r"Equivariance error $E_{\max}$", fontsize=9.5, labelpad=5)
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(f"Noise {noise}", fontsize=10.5, pad=6)
    # Shared ticks make comparisons across noises possible without repeated labels.
    axes[0, 0].yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.4, 0.8]))
    axes[0, 0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes[1, 0].yaxis.set_major_locator(FixedLocator([10.0**power for power in range(-7, 0)]))
    figure.legend(handles=[Patch(facecolor=COLORS[method], edgecolor="#333333", label=method) for method in METHODS],
                  loc="lower center", bbox_to_anchor=(0.5, 0.012), ncols=len(METHODS),
                  frameon=False, fontsize=9, handlelength=1.05, handletextpad=0.45, columnspacing=1.0)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = build_figure(linear_point(args.project_root))
    for suffix in ("pdf", "png"):
        figure.savefig(args.output_dir/f"linear_accuracy_equivariance_boxplots.{suffix}", dpi=350)
    plt.close(figure)


if __name__ == "__main__":
    main()
