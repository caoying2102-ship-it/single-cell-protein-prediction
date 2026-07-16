from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RIDGE_DIR = (
    ROOT
    / "results"
    / "EXP01_pca50_ridge"
)

COEFFICIENT_PATH = (
    RIDGE_DIR / "ridge_coefficients.csv"
)

PROTEIN_METRICS_PATH = (
    RIDGE_DIR / "per_protein_metrics.csv"
)

FIGURE_DIR = ROOT / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def is_pc_name(value: object) -> bool:
    return bool(
        re.fullmatch(
            r"PC[_\s-]?\d+",
            str(value),
            flags=re.IGNORECASE,
        )
    )


def pc_number(pc_name: str) -> int:
    match = re.search(r"\d+", str(pc_name))

    if match is None:
        return 9999

    return int(match.group())


def load_coefficients() -> pd.DataFrame:
    """
    Return a protein × PC coefficient matrix.
    Rows are proteins and columns are PC1 ... PC50.
    """

    raw = pd.read_csv(COEFFICIENT_PATH)
    lower_lookup = {
        column.lower(): column
        for column in raw.columns
    }

    # Case 1: long-format table
    long_required = {
        "protein",
        "feature",
        "coefficient",
    }

    if long_required.issubset(lower_lookup):
        coefficient_matrix = raw.pivot(
            index=lower_lookup["protein"],
            columns=lower_lookup["feature"],
            values=lower_lookup["coefficient"],
        )

    else:
        # Case 2: proteins are rows and PCs are columns
        pc_columns = [
            column
            for column in raw.columns
            if is_pc_name(column)
        ]

        if len(pc_columns) >= 2:
            non_pc_columns = [
                column
                for column in raw.columns
                if column not in pc_columns
            ]

            if len(non_pc_columns) == 0:
                raw.index = [
                    f"protein_{index}"
                    for index in range(len(raw))
                ]
                coefficient_matrix = raw[pc_columns]
            else:
                index_column = non_pc_columns[0]

                coefficient_matrix = (
                    raw.set_index(index_column)[pc_columns]
                )

        else:
            # Case 3: PCs are rows and proteins are columns
            first_column = raw.columns[0]

            pc_row_fraction = (
                raw[first_column]
                .astype(str)
                .map(is_pc_name)
                .mean()
            )

            if pc_row_fraction < 0.5:
                raise ValueError(
                    "Could not determine coefficient-table "
                    f"orientation. Columns: {list(raw.columns)}"
                )

            coefficient_matrix = (
                raw.set_index(first_column).transpose()
            )

    coefficient_matrix.index = (
        coefficient_matrix.index.astype(str)
    )

    coefficient_matrix.columns = [
        str(column).replace("_", "").replace("-", "")
        for column in coefficient_matrix.columns
    ]

    coefficient_matrix = coefficient_matrix.apply(
        pd.to_numeric,
        errors="coerce",
    )

    coefficient_matrix = coefficient_matrix.loc[
        :,
        [
            column
            for column in coefficient_matrix.columns
            if is_pc_name(column)
        ],
    ]

    coefficient_matrix = coefficient_matrix.dropna(
        axis=0,
        how="all",
    ).dropna(
        axis=1,
        how="all",
    )

    ordered_columns = sorted(
        coefficient_matrix.columns,
        key=pc_number,
    )

    return coefficient_matrix[ordered_columns]


def load_protein_metrics() -> pd.DataFrame:
    metrics = pd.read_csv(PROTEIN_METRICS_PATH)

    lookup = {
        column.lower(): column
        for column in metrics.columns
    }

    protein_column = next(
        (
            lookup[name]
            for name in [
                "protein",
                "protein_name",
                "target",
            ]
            if name in lookup
        ),
        None,
    )

    pcc_column = next(
        (
            lookup[name]
            for name in [
                "pcc",
                "pearson",
                "pearson_correlation",
            ]
            if name in lookup
        ),
        None,
    )

    if protein_column is None or pcc_column is None:
        raise ValueError(
            "Could not identify protein/PCC columns. "
            f"Columns: {list(metrics.columns)}"
        )

    return metrics[
        [protein_column, pcc_column]
    ].rename(
        columns={
            protein_column: "protein",
            pcc_column: "PCC",
        }
    )


def compute_global_importance(
    coefficients: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate coefficient magnitude across all proteins.

    Mean absolute coefficient is used because positive and
    negative coefficients would otherwise cancel.
    """

    importance = pd.DataFrame(
        {
            "PC": coefficients.columns,
            "mean_absolute_coefficient": (
                coefficients.abs().mean(axis=0).values
            ),
            "median_absolute_coefficient": (
                coefficients.abs().median(axis=0).values
            ),
            "coefficient_l2_norm": np.sqrt(
                np.square(coefficients).sum(axis=0)
            ).values,
        }
    )

    return importance.sort_values(
        "mean_absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)


def select_heatmap_proteins(
    coefficients: pd.DataFrame,
    protein_metrics: pd.DataFrame,
    number: int = 20,
) -> list[str]:
    """
    Select proteins with the highest Day 4 PCC so that the
    coefficient heatmap focuses on successfully predicted
    targets.
    """

    available = set(coefficients.index)

    protein_metrics = protein_metrics.copy()
    protein_metrics["protein"] = (
        protein_metrics["protein"].astype(str)
    )

    selected = (
        protein_metrics[
            protein_metrics["protein"].isin(available)
        ]
        .sort_values("PCC", ascending=False)
        .head(number)["protein"]
        .tolist()
    )

    if len(selected) < number:
        remaining = (
            coefficients.abs()
            .mean(axis=1)
            .sort_values(ascending=False)
            .index
        )

        for protein in remaining:
            if protein not in selected:
                selected.append(protein)

            if len(selected) == number:
                break

    return selected


def make_figure(
    coefficients: pd.DataFrame,
    importance: pd.DataFrame,
    protein_metrics: pd.DataFrame,
) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    top_pc_count = min(15, len(importance))
    top_pcs = importance.head(top_pc_count)["PC"].tolist()

    selected_proteins = select_heatmap_proteins(
        coefficients,
        protein_metrics,
        number=min(20, len(coefficients)),
    )

    heatmap_matrix = coefficients.loc[
        selected_proteins,
        top_pcs,
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6.2),
        gridspec_kw={"width_ratios": [0.7, 1.3]},
        constrained_layout=True,
    )

    # Panel A: global PC importance
    bar_data = (
        importance.head(top_pc_count)
        .sort_values(
            "mean_absolute_coefficient",
            ascending=True,
        )
    )

    axes[0].barh(
        bar_data["PC"],
        bar_data["mean_absolute_coefficient"],
        color="#0072B2",
        edgecolor="black",
        linewidth=0.4,
    )

    axes[0].set_xlabel(
        "Mean absolute Ridge coefficient"
    )
    axes[0].set_ylabel("PCA component")
    axes[0].set_title(
        "A. Global PCA feature importance",
        loc="left",
        fontweight="bold",
    )
    axes[0].grid(
        axis="x",
        alpha=0.2,
        linewidth=0.7,
    )

    # Panel B: signed coefficient heatmap
    values = heatmap_matrix.to_numpy()
    max_absolute_value = np.nanmax(np.abs(values))

    image = axes[1].imshow(
        values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-max_absolute_value,
        vmax=max_absolute_value,
        interpolation="nearest",
    )

    axes[1].set_xticks(
        np.arange(len(top_pcs))
    )
    axes[1].set_xticklabels(
        top_pcs,
        rotation=45,
        ha="right",
    )

    axes[1].set_yticks(
        np.arange(len(selected_proteins))
    )
    axes[1].set_yticklabels(
        selected_proteins,
        fontsize=8,
    )

    axes[1].set_xlabel("PCA component")
    axes[1].set_ylabel(
        "Proteins with highest Day 4 PCC"
    )
    axes[1].set_title(
        "B. Signed protein-specific coefficients",
        loc="left",
        fontweight="bold",
    )

    colorbar = fig.colorbar(
        image,
        ax=axes[1],
        fraction=0.035,
        pad=0.02,
    )
    colorbar.set_label("Ridge coefficient")

    fig.suptitle(
        "Interpretation of Ridge coefficients in PCA space",
        fontsize=14,
        fontweight="bold",
    )

    return fig


def print_diagnostics(
    coefficients: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    print("\nCoefficient matrix")
    print(
        f"Proteins × PCs: {coefficients.shape}"
    )

    print("\nTop 15 PCA components")
    print(
        importance.head(15).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print(
        "\nProteins with the largest absolute "
        "coefficients for the top five PCs"
    )

    for pc in importance.head(5)["PC"]:
        top_proteins = (
            coefficients[pc]
            .abs()
            .sort_values(ascending=False)
            .head(5)
            .index
        )

        print(f"\n{pc}")

        for protein in top_proteins:
            value = coefficients.loc[protein, pc]

            print(
                f"  {protein}: coefficient={value:.6f}"
            )


def main() -> None:
    coefficients = load_coefficients()
    protein_metrics = load_protein_metrics()

    importance = compute_global_importance(
        coefficients
    )

    importance_path = (
        RIDGE_DIR
        / "ridge_pca_importance_analysis.csv"
    )

    importance.to_csv(
        importance_path,
        index=False,
    )

    long_coefficients = (
        coefficients.reset_index()
        .rename(columns={"index": "protein"})
        .melt(
            id_vars="protein",
            var_name="PC",
            value_name="coefficient",
        )
    )

    long_path = (
        RIDGE_DIR
        / "ridge_coefficients_long.csv"
    )

    long_coefficients.to_csv(
        long_path,
        index=False,
    )

    fig = make_figure(
        coefficients,
        importance,
        protein_metrics,
    )

    png_path = (
        FIGURE_DIR
        / "figure4_ridge_pca_coefficients.png"
    )

    pdf_path = (
        FIGURE_DIR
        / "figure4_ridge_pca_coefficients.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    plt.close(fig)

    print_diagnostics(
        coefficients,
        importance,
    )

    print("\nSaved:")
    print(importance_path)
    print(long_path)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()