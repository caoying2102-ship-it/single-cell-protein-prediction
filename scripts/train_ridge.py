"""Train and evaluate the leak-free PCA + Ridge baseline."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from scprotein.metrics import summarize_predictions
from scprotein.models import (
    build_ridge_model,
    fit_ridge_model,
    predict_ridge_model,
)


EXPERIMENT_NAME = "EXP01_pca50_ridge"


def resolve_project_path(path_value):
    """Resolve a path relative to the project root."""
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_config(config_path):
    """Load YAML configuration."""
    config_path = resolve_project_path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config, config_path


def load_prepared_data(data_path):
    """Load the compact arrays produced by prepare_data.py."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Prepared data not found: {data_path}\n"
            "Run scripts/prepare_data.py first."
        )

    with np.load(data_path) as data:
        required_arrays = {
            "X_train",
            "X_validation",
            "y_train",
            "y_validation",
            "train_cell_ids",
            "validation_cell_ids",
            "feature_names",
            "protein_names",
        }

        missing_arrays = required_arrays.difference(
            data.files
        )

        if missing_arrays:
            raise ValueError(
                "Prepared archive is missing arrays: "
                f"{sorted(missing_arrays)}"
            )

        arrays = {
            name: data[name]
            for name in required_arrays
        }

    return arrays


def save_ridge_coefficients(
    model,
    protein_names,
    feature_names,
    output_directory,
):
    """Save standardized Ridge coefficients and intercepts."""
    ridge_model = model.named_steps["ridge"]

    coefficients = np.asarray(ridge_model.coef_)

    if coefficients.ndim == 1:
        coefficients = coefficients.reshape(1, -1)

    expected_shape = (
        len(protein_names),
        len(feature_names),
    )

    if coefficients.shape != expected_shape:
        raise ValueError(
            f"Unexpected coefficient shape "
            f"{coefficients.shape}; expected {expected_shape}."
        )

    coefficient_table = pd.DataFrame(
        coefficients,
        index=protein_names,
        columns=feature_names,
    )
    coefficient_table.index.name = "protein"

    coefficient_table.to_csv(
        output_directory / "ridge_coefficients.csv"
    )

    intercept_table = pd.DataFrame(
        {
            "protein": protein_names,
            "intercept": np.asarray(
                ridge_model.intercept_
            ).reshape(-1),
        }
    )
    intercept_table.to_csv(
        output_directory / "ridge_intercepts.csv",
        index=False,
    )

    global_importance = (
        coefficient_table
        .abs()
        .mean(axis=0)
        .sort_values(ascending=False)
        .rename("mean_absolute_coefficient")
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    global_importance.to_csv(
        output_directory
        / "ridge_global_feature_importance.csv",
        index=False,
    )


def main(config_path):
    config, resolved_config_path = load_config(
        config_path
    )

    seed = int(config["seed"])
    np.random.seed(seed)

    prepared_data_path = resolve_project_path(
        config["output"]["processed_dir"]
    ) / "cite_day_split_pca.npz"

    result_root = resolve_project_path(
        config["output"]["result_dir"]
    )
    model_root = resolve_project_path(
        config["output"]["model_dir"]
    )

    experiment_directory = (
        result_root / EXPERIMENT_NAME
    )
    experiment_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    model_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Prepared data: {prepared_data_path}")
    print(f"Results: {experiment_directory}")

    arrays = load_prepared_data(
        prepared_data_path
    )

    X_train = arrays["X_train"].astype(
        np.float32,
        copy=False,
    )
    X_validation = arrays["X_validation"].astype(
        np.float32,
        copy=False,
    )
    y_train = arrays["y_train"].astype(
        np.float32,
        copy=False,
    )
    y_validation = arrays["y_validation"].astype(
        np.float32,
        copy=False,
    )

    train_cell_ids = arrays[
        "train_cell_ids"
    ].astype(str)
    validation_cell_ids = arrays[
        "validation_cell_ids"
    ].astype(str)
    feature_names = arrays[
        "feature_names"
    ].astype(str)
    protein_names = arrays[
        "protein_names"
    ].astype(str)

    print("\nLoaded arrays")
    print(f"X_train:      {X_train.shape}")
    print(f"X_validation: {X_validation.shape}")
    print(f"y_train:      {y_train.shape}")
    print(f"y_validation: {y_validation.shape}")

    alpha = float(config["ridge"]["alpha"])

    model = build_ridge_model(alpha=alpha)

    start_time = time.perf_counter()

    model = fit_ridge_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
    )

    print("\nPredicting validation cells...")
    validation_predictions = predict_ridge_model(
        model,
        X_validation,
    ).astype(np.float32)

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    (
        metric_summary,
        protein_metrics,
        cell_correlations,
    ) = summarize_predictions(
        y_true=y_validation,
        y_pred=validation_predictions,
        protein_names=protein_names,
    )

    metric_summary.update(
        {
            "experiment": EXPERIMENT_NAME,
            "model": "Ridge",
            "alpha": alpha,
            "n_pca_components": int(
                X_train.shape[1]
            ),
            "train_days": config["split"][
                "train_days"
            ],
            "validation_days": config["split"][
                "validation_days"
            ],
            "training_seconds": float(
                elapsed_seconds
            ),
            "random_seed": seed,
            "pca_fit_on_training_only": True,
            "uses_target_derived_features": False,
        }
    )

    # Save predictions.
    np.save(
        experiment_directory
        / "validation_predictions.npy",
        validation_predictions,
    )

    np.save(
        experiment_directory
        / "validation_targets.npy",
        y_validation,
    )

    # Save per-protein metrics.
    protein_metrics.to_csv(
        experiment_directory
        / "per_protein_metrics.csv",
        index=False,
    )

    # Save per-cell official metric values.
    cell_metric_table = pd.DataFrame(
        {
            "cell_id": validation_cell_ids,
            "rowwise_pearson": cell_correlations,
        }
    )
    cell_metric_table.to_csv(
        experiment_directory
        / "per_cell_correlations.csv",
        index=False,
    )

    # Save coefficients for interpretation.
    save_ridge_coefficients(
        model=model,
        protein_names=protein_names,
        feature_names=feature_names,
        output_directory=experiment_directory,
    )

    # Save fitted scaler + Ridge model.
    model_path = (
        model_root / "pca50_ridge.joblib"
    )
    joblib.dump(model, model_path)

    # Save experiment configuration.
    shutil.copy2(
        resolved_config_path,
        experiment_directory / "config.yaml",
    )

    with (
        experiment_directory
        / "metrics_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            metric_summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nValidation results")
    print(
        "Kaggle-style row-wise Pearson: "
        f"{metric_summary['rowwise_pearson']:.6f}"
    )
    print(
        "Mean protein-wise PCC:          "
        f"{metric_summary['mean_proteinwise_pcc']:.6f}"
    )
    print(
        "Median protein-wise PCC:        "
        f"{metric_summary['median_proteinwise_pcc']:.6f}"
    )
    print(
        "RMSE:                           "
        f"{metric_summary['rmse']:.6f}"
    )
    print(
        "Training and prediction time:   "
        f"{elapsed_seconds:.2f} seconds"
    )

    print("\nTop 10 proteins by PCC")
    print(
        protein_metrics
        .sort_values("PCC", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print(f"\nModel saved to: {model_path}")
    print(
        "Experiment results saved to: "
        f"{experiment_directory}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train the leak-free PCA + Ridge baseline."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )

    arguments = parser.parse_args()
    main(arguments.config)