from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


EXPERIMENTS = [
    {
        "model": "Mean-profile",
        "directory": "EXP00_mean_profile",
        "color": "#8C8C8C",
    },
    {
        "model": "Ridge",
        "directory": "EXP01_pca50_ridge",
        "color": "#0072B2",
    },
    {
        "model": "Multi-output MLP",
        "directory": "EXP02_pca50_multioutput_mlp",
        "color": "#D55E00",
    },
]


def load_results() -> pd.DataFrame:
    rows = []

    for experiment in EXPERIMENTS:
        summary_path = (
            RESULTS_DIR
            / experiment["directory"]
            / "metrics_summary.json"
        )

        if not summary_path.exists():
            raise FileNotFoundError(
                f"Cannot find metrics file: {summary_path}"
            )

        with summary_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

        rows.append(
            {
                "model": experiment["model"],
                "experiment": experiment["directory"],
                "rowwise_pearson": metrics["rowwise_pearson"],
                "mean_proteinwise_pcc": metrics.get(
                    "mean_proteinwise_pcc"
                ),
                "median_proteinwise_pcc": metrics.get(
                    "median_proteinwise_pcc"
                ),
                "rmse": metrics["rmse"],
                "n_cells": metrics["n_cells"],
                "n_proteins": metrics["n_proteins"],
                "color": experiment["color"],
            }
        )

    return pd.DataFrame(rows)


def add_bar_labels(
    ax: plt.Axes,
    bars,
    values: np.ndarray,
    decimals: int = 3,
) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def make_figure(results: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )

    colors = results["color"].tolist()
    labels = results["model"].tolist()
    x = np.arange(len(results))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4.2),
        constrained_layout=True,
    )

    # Panel A: official competition-style metric
    rowwise_values = results["rowwise_pearson"].to_numpy()

    bars = axes[0].bar(
        x,
        rowwise_values,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        width=0.68,
    )

    axes[0].set_title(
        "A. Kaggle-style validation performance",
        loc="left",
        fontweight="bold",
    )
    axes[0].set_ylabel("Mean row-wise Pearson correlation")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylim(
        max(0, rowwise_values.min() - 0.08),
        min(1.0, rowwise_values.max() + 0.04),
    )
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.7)
    add_bar_labels(axes[0], bars, rowwise_values)

    # Panel B: absolute prediction error
    rmse_values = results["rmse"].to_numpy()

    bars = axes[1].bar(
        x,
        rmse_values,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        width=0.68,
    )

    axes[1].set_title(
        "B. Prediction error",
        loc="left",
        fontweight="bold",
    )
    axes[1].set_ylabel("RMSE ")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylim(0, rmse_values.max() * 1.18)
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.7)
    add_bar_labels(axes[1], bars, rmse_values)

    fig.suptitle(
        "Comparison of protein abundance prediction models",
        fontsize=14,
        fontweight="bold",
    )

    fig.text(
        0.5,
        -0.03,
        "Validation: Day 4 cells; training: Day 2–3 cells. "
        "Ridge and MLP use the same 50 PCA features.",
        ha="center",
        fontsize=9,
        color="#444444",
    )

    return fig


def print_summary(results: pd.DataFrame) -> None:
    baseline = results.loc[
        results["model"] == "Mean-profile"
    ].iloc[0]

    print("\nModel comparison")
    print(
        results[
            [
                "model",
                "rowwise_pearson",
                "mean_proteinwise_pcc",
                "rmse",
            ]
        ].to_string(index=False)
    )

    print("\nImprovement over mean-profile baseline")

    for model_name in ["Ridge", "Multi-output MLP"]:
        row = results.loc[results["model"] == model_name].iloc[0]

        pearson_gain = (
            row["rowwise_pearson"]
            - baseline["rowwise_pearson"]
        )

        rmse_reduction = (
            baseline["rmse"] - row["rmse"]
        ) / baseline["rmse"] * 100

        print(
            f"{model_name}: "
            f"row-wise Pearson +{pearson_gain:.4f}; "
            f"RMSE reduction {rmse_reduction:.1f}%"
        )


def main() -> None:
    results = load_results()

    comparison_path = RESULTS_DIR / "model_comparison.csv"
    results.drop(columns="color").to_csv(
        comparison_path,
        index=False,
    )

    fig = make_figure(results)

    png_path = (
        FIGURE_DIR / "figure1_model_comparison.png"
    )
    pdf_path = (
        FIGURE_DIR / "figure1_model_comparison.pdf"
    )

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print_summary(results)

    print("\nSaved:")
    print(comparison_path)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()