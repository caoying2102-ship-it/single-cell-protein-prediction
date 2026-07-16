from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RIDGE_PATH = (
    RESULTS_DIR
    / "EXP01_pca50_ridge"
    / "per_protein_metrics.csv"
)

MLP_PATH = (
    RESULTS_DIR
    / "EXP02_pca50_multioutput_mlp"
    / "per_protein_metrics.csv"
)


def standardize_columns(
    data: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """Identify protein and PCC columns robustly."""

    column_lookup = {
        column.lower(): column
        for column in data.columns
    }

    protein_candidates = [
        "protein",
        "protein_name",
        "target",
    ]
    pcc_candidates = [
        "pcc",
        "pearson",
        "pearson_correlation",
    ]

    protein_column = next(
        (
            column_lookup[name]
            for name in protein_candidates
            if name in column_lookup
        ),
        None,
    )

    pcc_column = next(
        (
            column_lookup[name]
            for name in pcc_candidates
            if name in column_lookup
        ),
        None,
    )

    if protein_column is None or pcc_column is None:
        raise ValueError(
            f"Unexpected columns in {model_name}: "
            f"{list(data.columns)}"
        )

    return data[
        [protein_column, pcc_column]
    ].rename(
        columns={
            protein_column: "protein",
            pcc_column: f"{model_name}_pcc",
        }
    )


def load_results() -> pd.DataFrame:
    ridge = standardize_columns(
        pd.read_csv(RIDGE_PATH),
        "ridge",
    )

    mlp = standardize_columns(
        pd.read_csv(MLP_PATH),
        "mlp",
    )

    paired = ridge.merge(
        mlp,
        on="protein",
        how="inner",
        validate="one_to_one",
    )

    paired = paired.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(
        subset=["ridge_pcc", "mlp_pcc"]
    )

    paired["pcc_difference"] = (
        paired["mlp_pcc"] - paired["ridge_pcc"]
    )

    tolerance = 1e-6

    paired["better_model"] = np.select(
        [
            paired["pcc_difference"] > tolerance,
            paired["pcc_difference"] < -tolerance,
        ],
        [
            "MLP higher",
            "Ridge higher",
        ],
        default="Approximately tied",
    )

    return paired


def make_figure(paired: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.6),
        gridspec_kw={"width_ratios": [1.15, 0.85]},
        constrained_layout=True,
    )

    styles = {
        "MLP higher": {
            "color": "#D55E00",
            "marker": "o",
            "label": "MLP higher",
        },
        "Ridge higher": {
            "color": "#0072B2",
            "marker": "^",
            "label": "Ridge higher",
        },
        "Approximately tied": {
            "color": "#8C8C8C",
            "marker": "s",
            "label": "Approximately tied",
        },
    }

    # Panel A: paired PCC scatter plot
    all_values = np.concatenate(
        [
            paired["ridge_pcc"].to_numpy(),
            paired["mlp_pcc"].to_numpy(),
        ]
    )

    lower = min(-0.1, np.nanmin(all_values) - 0.05)
    upper = min(1.0, np.nanmax(all_values) + 0.05)

    for group_name, style in styles.items():
        subset = paired[
            paired["better_model"] == group_name
        ]

        if subset.empty:
            continue

        axes[0].scatter(
            subset["ridge_pcc"],
            subset["mlp_pcc"],
            s=35,
            alpha=0.75,
            color=style["color"],
            marker=style["marker"],
            edgecolor="white",
            linewidth=0.4,
            label=f"{style['label']} (n={len(subset)})",
        )

    axes[0].plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        color="black",
        linewidth=1,
        label="Equal performance",
    )

    axes[0].set_xlim(lower, upper)
    axes[0].set_ylim(lower, upper)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("Ridge protein-wise PCC")
    axes[0].set_ylabel("MLP protein-wise PCC")
    axes[0].set_title(
        "A. Paired protein-level performance",
        loc="left",
        fontweight="bold",
    )
    axes[0].grid(alpha=0.2, linewidth=0.7)
    axes[0].legend(
        frameon=False,
        fontsize=8,
        loc="lower right",
    )

    # Label proteins with largest absolute disagreement
    most_different = paired.reindex(
        paired["pcc_difference"]
        .abs()
        .sort_values(ascending=False)
        .head(5)
        .index
    )

    for _, row in most_different.iterrows():
        axes[0].annotate(
            row["protein"],
            xy=(row["ridge_pcc"], row["mlp_pcc"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
            color="#333333",
        )

    # Panel B: distribution of paired differences
    differences = paired["pcc_difference"]

    axes[1].hist(
        differences,
        bins=24,
        color="#6A8CAF",
        edgecolor="white",
        linewidth=0.7,
        alpha=0.9,
    )

    axes[1].axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Equal performance",
    )

    axes[1].axvline(
        differences.mean(),
        color="#D55E00",
        linewidth=1.5,
        label=f"Mean difference = {differences.mean():.3f}",
    )

    axes[1].set_xlabel("MLP PCC − Ridge PCC")
    axes[1].set_ylabel("Number of proteins")
    axes[1].set_title(
        "B. Distribution of paired differences",
        loc="left",
        fontweight="bold",
    )
    axes[1].grid(axis="y", alpha=0.2, linewidth=0.7)
    axes[1].legend(frameon=False, fontsize=8)

    figure_title = (
        "Ridge and MLP show similar protein-level "
        "predictive performance"
    )

    fig.suptitle(
        figure_title,
        fontsize=14,
        fontweight="bold",
    )

    return fig


def print_summary(paired: pd.DataFrame) -> None:
    ridge_wins = int(
        (paired["better_model"] == "Ridge higher").sum()
    )
    mlp_wins = int(
        (paired["better_model"] == "MLP higher").sum()
    )
    ties = int(
        (
            paired["better_model"]
            == "Approximately tied"
        ).sum()
    )

    correlation = np.corrcoef(
        paired["ridge_pcc"],
        paired["mlp_pcc"],
    )[0, 1]

    print("\nPaired protein-level comparison")
    print(f"Proteins compared: {len(paired)}")
    print(f"Ridge higher: {ridge_wins}")
    print(f"MLP higher: {mlp_wins}")
    print(f"Approximately tied: {ties}")
    print(
        "Mean MLP − Ridge PCC: "
        f"{paired['pcc_difference'].mean():.6f}"
    )
    print(
        "Median MLP − Ridge PCC: "
        f"{paired['pcc_difference'].median():.6f}"
    )
    print(
        "Correlation between Ridge and MLP PCC: "
        f"{correlation:.4f}"
    )

    print("\nProteins with largest absolute differences:")
    largest = paired.reindex(
        paired["pcc_difference"]
        .abs()
        .sort_values(ascending=False)
        .head(10)
        .index
    )

    print(
        largest[
            [
                "protein",
                "ridge_pcc",
                "mlp_pcc",
                "pcc_difference",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    paired = load_results()

    table_path = (
        RESULTS_DIR
        / "ridge_mlp_per_protein_comparison.csv"
    )

    paired.sort_values(
        "pcc_difference",
        ascending=False,
    ).to_csv(
        table_path,
        index=False,
    )

    fig = make_figure(paired)

    png_path = (
        FIGURE_DIR
        / "figure2_ridge_mlp_per_protein_pcc.png"
    )
    pdf_path = (
        FIGURE_DIR
        / "figure2_ridge_mlp_per_protein_pcc.pdf"
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

    print_summary(paired)

    print("\nSaved:")
    print(table_path)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()