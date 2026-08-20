"""Compare fixed-reference OTRR with the matched semi-synthetic baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPRESENTATIONS = ("identity", "rotation_2026081401", "rotation_2026081402")
METHOD_FILES = {
    "LS": "ls.npz",
    "Radial Huber": "radial_huber.npz",
    "CW-Huber": "cw_huber.npz",
    "CW-ASM": "cw_asm.npz",
    "CW-CQR": "cw_cqr.npz",
}
KAPPAS = (0.25, 0.5)
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 20260820


def _latex_value(mean: float, sd: float) -> str:
    return f"{mean:.3f} ({sd:.3f})"


def write_latex_table(summary: pd.DataFrame, output: Path) -> None:
    """Write the paper-facing Supplementary Table S2.4 from the final summary."""

    rows: list[str] = []
    for row in summary.itertuples(index=False):
        rows.append(
            f"{row.kappa:.2f} & {row.method} & "
            f"{_latex_value(row.tpr_mean, row.tpr_sd)} & "
            f"{_latex_value(row.fdr_mean, row.fdr_sd)} & "
            f"{_latex_value(row.iou_mean, row.iou_sd)} & "
            f"{row.average_intersection:.1f}/{row.average_union:.1f} "
            f"({row.ratio_bootstrap_sd:.3f}) \\\\"
        )
    (output / "table_s2_4_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    table = [
        r"\begin{tabular}{clcccc}",
        r"\hline",
        r"$\kappa$ & Method & TPR & FDR & IoU & Avg. intersection / avg. union \\",
        r"\hline",
        *rows,
        r"\hline",
        r"\end{tabular}",
    ]
    (output / "table_s2_4.tex").write_text("\n".join(table) + "\n", encoding="utf-8")


def _safe_fdr(selected: np.ndarray, truth: np.ndarray) -> float:
    discoveries = int(selected.sum())
    return float(np.logical_and(selected, ~truth).sum() / discoveries) if discoveries else 0.0


def _record(
    *,
    method: str,
    kappa: float,
    repetition: int,
    representation: str,
    selected: np.ndarray,
    truth: np.ndarray,
    provenance: str,
) -> dict[str, object]:
    selected = np.asarray(selected, dtype=bool).reshape(-1)
    truth = np.asarray(truth, dtype=bool).reshape(-1)
    if selected.shape != truth.shape:
        raise ValueError(f"selection/truth shape mismatch: {selected.shape} != {truth.shape}")
    tp = int(np.logical_and(selected, truth).sum())
    fp = int(np.logical_and(selected, ~truth).sum())
    return {
        "noise": "C",
        "kappa": kappa,
        "repetition": repetition,
        "representation": representation,
        "method": method,
        "selected": int(selected.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "tpr": float(tp / truth.sum()),
        "fdr": _safe_fdr(selected, truth),
        "data_provenance": provenance,
        "paired_comparable_to_baselines": method != "OTRR",
    }


def baseline_records(root: Path) -> tuple[list[dict[str, object]], np.ndarray]:
    with np.load(root / "prepared" / "template.npz", allow_pickle=False) as state:
        truth = np.asarray(state["stable_mask"], dtype=bool).reshape(-1)
    records: list[dict[str, object]] = []
    for kappa in KAPPAS:
        for repetition in range(1, 11):
            task = root / "tasks" / f"rep{repetition:03d}_noiseC_kappa{kappa:g}"
            for representation in REPRESENTATIONS:
                for method, filename in METHOD_FILES.items():
                    with np.load(task / representation / filename, allow_pickle=False) as state:
                        selected = np.asarray(state["selected"], dtype=bool)
                    records.append(
                        _record(
                            method=method,
                            kappa=kappa,
                            repetition=repetition,
                            representation=representation,
                            selected=selected,
                            truth=truth,
                            provenance="mac_original_Y",
                        )
                    )
    return records, truth


def otrr_records(root: Path, truth: np.ndarray) -> list[dict[str, object]]:
    contrasts = pd.read_csv(root / "genus_contrasts.csv")
    records: list[dict[str, object]] = []
    for (kappa, repetition, representation), group in contrasts.groupby(
        ["kappa", "rep", "representation"], sort=True
    ):
        ordered = group.sort_values(["genus_index", "predictor_index"])
        observed_truth = ordered["truth_nonzero"].astype(bool).to_numpy()
        if not np.array_equal(observed_truth, truth):
            raise ValueError(f"OTRR truth mask mismatch for kappa={kappa}, rep={repetition}, {representation}")
        records.append(
            _record(
                method="OTRR",
                kappa=float(kappa),
                repetition=int(repetition),
                representation=str(representation),
                selected=ordered["selected_bh_q05"].astype(bool).to_numpy(),
                truth=truth,
                provenance="linux_reconstructed_Y",
            )
        )
    return records


def repetition_metrics(representation_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["noise", "kappa", "repetition", "method", "data_provenance"]
    for key, group in representation_metrics.groupby(keys, sort=True):
        if set(group["representation"]) != set(REPRESENTATIONS):
            raise ValueError(f"incomplete representation set for {key}")
        # Recover the sets from source artifacts through a stable integer key stored below.
        source_sets = group.attrs.get("selection_sets")
        if source_sets is None:
            raise RuntimeError("selection sets were not attached to the representation metrics")
        selected_sets = [source_sets[(key[1], key[2], key[3], representation)] for representation in REPRESENTATIONS]
        intersection = np.logical_and.reduce(selected_sets)
        union = np.logical_or.reduce(selected_sets)
        rows.append(
            {
                "noise": key[0],
                "kappa": key[1],
                "repetition": key[2],
                "method": key[3],
                "data_provenance": key[4],
                "tpr": float(group["tpr"].mean()),
                "fdr": float(group["fdr"].mean()),
                "intersection": int(intersection.sum()),
                "union": int(union.sum()),
                "iou": float(intersection.sum() / union.sum()) if union.any() else 1.0,
            }
        )
    return pd.DataFrame(rows)


def ratio_of_averages_bootstrap(repetition: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap the ratio of average intersection to average union by repetition."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []
    for key, group in repetition.groupby(
        ["noise", "kappa", "method", "data_provenance"], sort=True
    ):
        group = group.sort_values("repetition")
        intersection = group["intersection"].to_numpy(dtype=float)
        union = group["union"].to_numpy(dtype=float)
        n = len(group)
        indices = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
        bootstrap_intersection = intersection[indices].mean(axis=1)
        bootstrap_union = union[indices].mean(axis=1)
        bootstrap_ratio = np.divide(
            bootstrap_intersection,
            bootstrap_union,
            out=np.ones_like(bootstrap_intersection),
            where=bootstrap_union != 0,
        )
        average_intersection = float(intersection.mean())
        average_union = float(union.mean())
        rows.append(
            {
                "noise": key[0],
                "kappa": key[1],
                "method": key[2],
                "data_provenance": key[3],
                "average_intersection": average_intersection,
                "average_union": average_union,
                "ratio_of_averages": (
                    average_intersection / average_union if average_union else 1.0
                ),
                "ratio_bootstrap_sd": float(bootstrap_ratio.std(ddof=1)),
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--otrr-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline, truth = baseline_records(args.baseline_root)
    otrr = otrr_records(args.otrr_root, truth)
    records = baseline + otrr
    representation = pd.DataFrame(records)

    # Keep the exact selected sets out of CSV while making them available to the
    # repetition-level aggregation.
    selection_sets: dict[tuple[float, int, str, str], np.ndarray] = {}
    for kappa in KAPPAS:
        for repetition in range(1, 11):
            task = args.baseline_root / "tasks" / f"rep{repetition:03d}_noiseC_kappa{kappa:g}"
            for representation_name in REPRESENTATIONS:
                for method, filename in METHOD_FILES.items():
                    with np.load(task / representation_name / filename, allow_pickle=False) as state:
                        selection_sets[(kappa, repetition, method, representation_name)] = np.asarray(
                            state["selected"], dtype=bool
                        ).reshape(-1)
    contrasts = pd.read_csv(args.otrr_root / "genus_contrasts.csv")
    for (kappa, repetition, representation_name), group in contrasts.groupby(
        ["kappa", "rep", "representation"], sort=True
    ):
        selection_sets[(float(kappa), int(repetition), "OTRR", str(representation_name))] = (
            group.sort_values(["genus_index", "predictor_index"])["selected_bh_q05"].astype(bool).to_numpy()
        )
    representation.attrs["selection_sets"] = selection_sets
    repetition = repetition_metrics(representation)
    ratio_bootstrap = ratio_of_averages_bootstrap(repetition)

    representation_summary = (
        representation.groupby(
            ["noise", "kappa", "representation", "method", "data_provenance"], as_index=False
        )
        .agg(
            repetitions=("repetition", "count"),
            selected_mean=("selected", "mean"),
            selected_sd=("selected", "std"),
            tpr_mean=("tpr", "mean"),
            tpr_sd=("tpr", "std"),
            fdr_mean=("fdr", "mean"),
            fdr_sd=("fdr", "std"),
        )
        .sort_values(["kappa", "method", "representation"])
    )

    summary = (
        repetition.groupby(["noise", "kappa", "method", "data_provenance"], as_index=False)
        .agg(
            repetitions=("repetition", "count"),
            tpr_mean=("tpr", "mean"),
            tpr_sd=("tpr", "std"),
            fdr_mean=("fdr", "mean"),
            fdr_sd=("fdr", "std"),
            iou_mean=("iou", "mean"),
            iou_sd=("iou", "std"),
            intersection_mean=("intersection", "mean"),
            intersection_sd=("intersection", "std"),
            union_mean=("union", "mean"),
            union_sd=("union", "std"),
        )
        .sort_values(["kappa", "method"])
    )
    for metric in ("tpr", "fdr", "iou", "intersection", "union"):
        summary[f"{metric}_mcse"] = summary[f"{metric}_sd"] / np.sqrt(summary["repetitions"])
    summary["strictly_paired_comparison"] = False
    summary["comparison_note"] = (
        "Baselines use the serialized response; OTRR uses an independent reconstruction from the same source, configuration, and seed."
    )
    summary = summary.merge(
        ratio_bootstrap,
        on=["noise", "kappa", "method", "data_provenance"],
        validate="one_to_one",
    )

    args.output.mkdir(parents=True, exist_ok=True)
    representation.to_csv(args.output / "three_representation_metrics.csv", index=False)
    representation_summary.to_csv(args.output / "representation_summary.csv", index=False)
    repetition.to_csv(args.output / "repetition_metrics.csv", index=False)
    ratio_bootstrap.to_csv(args.output / "ratio_of_average_sets_bootstrap.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    write_latex_table(summary, args.output)

    report_lines = [
        "# Noise-C fixed-reference semi-synthetic comparison",
        "",
        "This comparison uses only the identity representation and rotations "
        "`2026081401` and `2026081402` for every method.",
        "",
        "## Provenance boundary",
        "",
        "The baseline artifacts use the original serialized response. The OTRR handoff omitted "
        "the response and reconstructed it independently from the same source, configuration, and seed. "
        "The response hashes differ, so cross-method TPR/FDR differences are descriptive rather than "
        "strictly paired. Within-method representation stability remains directly interpretable.",
        "",
        "## Repetition-level means",
        "",
        "| kappa | Method | TPR | FDR | Avg. intersection / avg. union | Three-representation IoU |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        report_lines.append(
            f"| {row.kappa:g} | {row.method} | {row.tpr_mean:.3f} ({row.tpr_sd:.3f}) | "
            f"{row.fdr_mean:.3f} ({row.fdr_sd:.3f}) | "
            f"{row.average_intersection:.1f}/{row.average_union:.1f} "
            f"({row.ratio_bootstrap_sd:.3f}) | "
            f"{row.iou_mean:.3f} ({row.iou_sd:.3f}) |"
        )
    report_lines.extend(
        [
            "",
            "TPR, FDR, and IoU parentheses are standard deviations across the ten repetitions. "
            "For the average-intersection/average-union column, parentheses give the paired "
            "repetition-bootstrap standard deviation of that ratio (100,000 draws; seed 20260820). "
            "The IoU is computed within each repetition and then averaged.",
            "",
        ]
    )
    (args.output / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
