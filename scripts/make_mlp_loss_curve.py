from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_DIR = (
    ROOT
    / "results"
    / "EXP02_pca50_multioutput_mlp"
)

HISTORY_PATH = EXPERIMENT_DIR / "training_history.csv"
SUMMARY_PATH = EXPERIMENT_DIR / "metrics_summary.json"

FIGURE_DIR = ROOT / "results" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def find_column(
    data: pd.DataFrame,
    candidates: list[str],
) -> str:
    """Find a column using case-insensitive candidates."""

    lookup = {
        column.lower(): column
        for column in data.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise ValueError(
        f"Could not find any of {candidates}. "
        f"Available columns: {list(data.columns)}"
    )


def load_history() -> pd.DataFrame:
    history = pd.read_csv(HISTORY_PATH)

    epoch_column = find_column(
        history,
        ["epoch", "epochs"],
    )

    train_column = find_column(
        history,
        [
            "train_loss",
            "training_loss",
            "train",
        ],
    )

    
    validation_column = find_column(
    history,
    [
        "internal_validation_loss",
        "internal_val_loss",
        "validation_loss",
        "val_loss",
        "internal_val",
    ],
    )
    history = history[
        [
            epoch_column,
            train_column,
            validation_column,
        ]
    ].rename(
        columns={
            epoch_column: "epoch",
            train_column: "train_loss",
            validation_column: "internal_val_loss",
        }
    )

    history = history.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    history = history.sort_values("epoch").reset_index(
        drop=True
    )

    history["generalization_gap"] = (
        history["internal_val_loss"]
        - history["train_loss"]
    )

    return history


def load_reported_best_epoch() -> int | None:
    if not SUMMARY_PATH.exists():
        return None

    with SUMMARY_PATH.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    best_epoch = summary.get("best_epoch")

    if best_epoch is None:
        return None

    return int(best_epoch)


def make_figure(
    history: pd.DataFrame,
    best_epoch: int,
) -> plt.Figure:
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
        figsize=(10.5, 4.2),
        gridspec_kw={"width_ratios": [1.35, 0.65]},
        constrained_layout=True,
    )

    # Locate best epoch in the stored history
    best_row = history.loc[
        history["epoch"] == best_epoch
    ]

    if best_row.empty:
        best_index = history[
            "internal_val_loss"
        ].idxmin()
        best_row = history.loc[[best_index]]
        best_epoch = int(best_row.iloc[0]["epoch"])

    best_validation_loss = float(
        best_row.iloc[0]["internal_val_loss"]
    )

    # Panel A: training and validation losses
    axes[0].plot(
        history["epoch"],
        history["train_loss"],
        color="#0072B2",
        linewidth=2,
        label="Training loss",
    )

    axes[0].plot(
        history["epoch"],
        history["internal_val_loss"],
        color="#D55E00",
        linewidth=2,
        label="Internal-validation loss",
    )

    axes[0].axvline(
        best_epoch,
        color="#333333",
        linestyle="--",
        linewidth=1,
    )

    axes[0].scatter(
        [best_epoch],
        [best_validation_loss],
        color="#D55E00",
        edgecolor="black",
        linewidth=0.6,
        s=50,
        zorder=5,
    )

    axes[0].annotate(
        (
            f"Best stored epoch: {best_epoch}\n"
            f"Validation loss: "
            f"{best_validation_loss:.4f}"
        ),
        xy=(best_epoch, best_validation_loss),
        xytext=(-95, 28),
        textcoords="offset points",
        fontsize=8.5,
        arrowprops={
            "arrowstyle": "->",
            "color": "#555555",
            "linewidth": 0.8,
        },
    )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Standardized MSE loss")
    axes[0].set_title(
        "A. Multi-output MLP learning curves",
        loc="left",
        fontweight="bold",
    )
    axes[0].grid(alpha=0.2, linewidth=0.7)
    axes[0].legend(frameon=False)

    # Panel B: validation minus training loss
    axes[1].plot(
        history["epoch"],
        history["generalization_gap"],
        color="#6A51A3",
        linewidth=2,
    )

    axes[1].axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    axes[1].axvline(
        best_epoch,
        color="#333333",
        linestyle="--",
        linewidth=1,
    )

    axes[1].fill_between(
        history["epoch"],
        0,
        history["generalization_gap"],
        color="#6A51A3",
        alpha=0.15,
    )

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation − training loss")
    axes[1].set_title(
        "B. Generalization gap",
        loc="left",
        fontweight="bold",
    )
    axes[1].grid(alpha=0.2, linewidth=0.7)

    fig.suptitle(
        "Training dynamics of the multi-output MLP",
        fontsize=14,
        fontweight="bold",
    )

    return fig


def print_diagnostics(
    history: pd.DataFrame,
    best_epoch: int,
) -> None:
    first = history.iloc[0]
    last = history.iloc[-1]

    minimum_index = history[
        "internal_val_loss"
    ].idxmin()

    minimum_row = history.loc[minimum_index]
    minimum_epoch = int(minimum_row["epoch"])
    maximum_epoch = int(history["epoch"].max())

    train_reduction = (
        first["train_loss"] - last["train_loss"]
    ) / first["train_loss"] * 100

    validation_reduction = (
        first["internal_val_loss"]
        - last["internal_val_loss"]
    ) / first["internal_val_loss"] * 100

    print("\nMLP training diagnostics")
    print(f"Epochs stored: {len(history)}")
    print(f"Maximum epoch: {maximum_epoch}")
    print(f"Reported best epoch: {best_epoch}")
    print(f"Minimum-loss epoch: {minimum_epoch}")
    print(
        f"Initial training loss: "
        f"{first['train_loss']:.6f}"
    )
    print(
        f"Final training loss: "
        f"{last['train_loss']:.6f}"
    )
    print(
        f"Initial internal-validation loss: "
        f"{first['internal_val_loss']:.6f}"
    )
    print(
        f"Final internal-validation loss: "
        f"{last['internal_val_loss']:.6f}"
    )
    print(
        f"Training-loss reduction: "
        f"{train_reduction:.2f}%"
    )
    print(
        f"Validation-loss reduction: "
        f"{validation_reduction:.2f}%"
    )
    print(
        f"Final generalization gap: "
        f"{last['generalization_gap']:.6f}"
    )

    if minimum_epoch == maximum_epoch:
        print(
            "\nInterpretation: the lowest validation loss "
            "occurred at the final allowed epoch. Training "
            "was stopped by the epoch limit rather than by "
            "early stopping."
        )
        print(
            "For the report, describe this MLP as a "
            "fixed-compute nonlinear baseline, not as a "
            "fully optimized neural-network model."
        )
    else:
        print(
            "\nInterpretation: validation loss reached its "
            "minimum before the maximum epoch, consistent "
            "with convergence or early stopping."
        )


def main() -> None:
    history = load_history()

    reported_best_epoch = load_reported_best_epoch()

    if reported_best_epoch is None:
        reported_best_epoch = int(
            history.loc[
                history["internal_val_loss"].idxmin(),
                "epoch",
            ]
        )

    fig = make_figure(
        history,
        reported_best_epoch,
    )

    png_path = (
        FIGURE_DIR
        / "figure3_mlp_training_curve.png"
    )
    pdf_path = (
        FIGURE_DIR
        / "figure3_mlp_training_curve.pdf"
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
        history,
        reported_best_epoch,
    )

    print("\nSaved:")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()