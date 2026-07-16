"""Leak-free feature engineering utilities."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def validate_feature_matrix(X, matrix_name):
    """Check that a feature matrix is valid for model fitting."""
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(
            f"{matrix_name} must be two-dimensional; "
            f"received shape {X.shape}."
        )

    if X.shape[0] == 0:
        raise ValueError(f"{matrix_name} contains no cells.")

    if X.shape[1] == 0:
        raise ValueError(f"{matrix_name} contains no features.")

    if not np.isfinite(X).all():
        n_invalid = np.size(X) - np.isfinite(X).sum()
        raise ValueError(
            f"{matrix_name} contains {n_invalid} NaN or infinite values."
        )


def fit_pca_on_training_data(
    X_train,
    X_validation,
    n_components=50,
    solver="randomized",
    random_state=42,
):
    """
    Fit PCA on training cells only.

    The validation matrix is transformed using the PCA model fitted on
    the training matrix. Validation cells are never used during fitting.
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    X_validation = np.asarray(
        X_validation,
        dtype=np.float32,
    )

    validate_feature_matrix(X_train, "X_train")
    validate_feature_matrix(X_validation, "X_validation")

    if X_train.shape[1] != X_validation.shape[1]:
        raise ValueError(
            "Training and validation matrices contain different "
            f"numbers of genes: {X_train.shape[1]} and "
            f"{X_validation.shape[1]}."
        )

    maximum_components = min(X_train.shape)

    if n_components > maximum_components:
        raise ValueError(
            f"n_components={n_components} exceeds the maximum "
            f"possible value {maximum_components}."
        )

    print("\nPCA configuration")
    print(f"Training matrix:       {X_train.shape}")
    print(f"Validation matrix:     {X_validation.shape}")
    print(f"Number of components:  {n_components}")
    print(f"SVD solver:            {solver}")
    print(f"Random state:          {random_state}")

    pca = PCA(
        n_components=n_components,
        svd_solver=solver,
        random_state=random_state,
    )

    # Important: only training cells are used here.
    print("\nFitting PCA on training cells only...")
    X_train_pca = pca.fit_transform(X_train)

    print("Transforming validation cells...")
    X_validation_pca = pca.transform(X_validation)

    X_train_pca = X_train_pca.astype(
        np.float32,
        copy=False,
    )
    X_validation_pca = X_validation_pca.astype(
        np.float32,
        copy=False,
    )

    feature_names = [
        f"PC{component_number}"
        for component_number in range(1, n_components + 1)
    ]

    total_explained_variance = float(
        pca.explained_variance_ratio_.sum()
    )

    print("\nPCA completed")
    print(f"Training PCA matrix:   {X_train_pca.shape}")
    print(f"Validation PCA matrix: {X_validation_pca.shape}")
    print(
        "Total explained variance ratio: "
        f"{total_explained_variance:.4f}"
    )

    return (
        X_train_pca,
        X_validation_pca,
        pca,
        feature_names,
    )


def create_pca_variance_table(pca, feature_names):
    """Create a table containing PCA variance information."""
    if len(feature_names) != len(
        pca.explained_variance_ratio_
    ):
        raise ValueError(
            "The number of feature names does not match "
            "the number of PCA components."
        )

    explained_variance = pca.explained_variance_ratio_

    variance_table = pd.DataFrame(
        {
            "feature": feature_names,
            "explained_variance_ratio": explained_variance,
            "cumulative_explained_variance": np.cumsum(
                explained_variance
            ),
        }
    )

    return variance_table


def save_pca_artifacts(
    pca,
    feature_names,
    output_directory,
):
    """Save the fitted PCA transformer and variance table."""
    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = output_directory / "pca_model.joblib"
    variance_path = (
        output_directory / "pca_explained_variance.csv"
    )
    feature_name_path = (
        output_directory / "pca_feature_names.txt"
    )

    joblib.dump(pca, model_path)

    variance_table = create_pca_variance_table(
        pca,
        feature_names,
    )
    variance_table.to_csv(
        variance_path,
        index=False,
    )

    feature_name_path.write_text(
        "\n".join(feature_names) + "\n",
        encoding="utf-8",
    )

    print("\nPCA artifacts saved")
    print(f"PCA model:             {model_path}")
    print(f"Variance table:        {variance_path}")
    print(f"Feature names:         {feature_name_path}")