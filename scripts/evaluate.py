"""Evaluate the no-input mean-profile baseline."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from scprotein.metrics import summarize_predictions


EXPERIMENT_NAME = "EXP00_mean_profile"


def main():
    data_path = (
        PROJECT_ROOT
        / "data/processed/cite_day_split_pca.npz"
    )
    result_directory = (
        PROJECT_ROOT
        / "results"
        / EXPERIMENT_NAME
    )
    result_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    with np.load(data_path) as data:
        y_train = data["y_train"].astype(
            np.float32,
            copy=False,
        )
        y_validation = data[
            "y_validation"
        ].astype(
            np.float32,
            copy=False,
        )
        validation_cell_ids = data[
            "validation_cell_ids"
        ].astype(str)
        protein_names = data[
            "protein_names"
        ].astype(str)

    # Optimal constant prediction under squared-error loss.
    mean_profile = y_train.mean(
        axis=0,
        keepdims=True,
    )

    predictions = np.repeat(
        mean_profile,
        repeats=y_validation.shape[0],
        axis=0,
    )

    (
        summary,
        protein_metrics,
        cell_correlations,
    ) = summarize_predictions(
        y_true=y_validation,
        y_pred=predictions,
        protein_names=protein_names,
    )

    summary.update(
        {
            "experiment": EXPERIMENT_NAME,
            "model": "Training-set mean protein profile",
            "uses_rna": False,
            "uses_metadata": False,
            "train_days": [2, 3],
            "validation_days": [4],
        }
    )

    with (
        result_directory / "metrics_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    pd.DataFrame(
        {
            "protein": protein_names,
            "training_mean": mean_profile.reshape(-1),
        }
    ).to_csv(
        result_directory / "mean_profile.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "cell_id": validation_cell_ids,
            "rowwise_pearson": cell_correlations,
        }
    ).to_csv(
        result_directory
        / "per_cell_correlations.csv",
        index=False,
    )

    protein_metrics.to_csv(
        result_directory
        / "per_protein_metrics.csv",
        index=False,
    )

    np.save(
        result_directory / "mean_profile.npy",
        mean_profile,
    )

    print("Mean-profile baseline")
    print(
        "Row-wise Pearson: "
        f"{summary['rowwise_pearson']:.6f}"
    )
    print(f"RMSE: {summary['rmse']:.6f}")
    print(
        "Mean protein-wise PCC: "
        f"{summary['mean_proteinwise_pcc']}"
    )
    print(f"Results: {result_directory}")


if __name__ == "__main__":
    main()