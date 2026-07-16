"""Evaluation metrics for multimodal protein prediction."""

import numpy as np
import pandas as pd


def validate_prediction_arrays(y_true, y_pred):
    """Validate prediction and target arrays."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, "
            f"y_pred={y_pred.shape}"
        )

    if y_true.ndim != 2:
        raise ValueError(
            "Targets and predictions must be two-dimensional."
        )

    if not np.isfinite(y_true).all():
        raise ValueError("y_true contains NaN or infinite values.")

    if not np.isfinite(y_pred).all():
        raise ValueError("y_pred contains NaN or infinite values.")

    return y_true, y_pred


def rowwise_pearson(y_true, y_pred, epsilon=1e-12):
    """
    Compute the Kaggle-style row-wise Pearson correlation.

    For each cell, correlation is calculated across all proteins.
    The final score is the mean correlation across valid cells.
    """
    y_true, y_pred = validate_prediction_arrays(
        y_true,
        y_pred,
    )

    true_centered = y_true - y_true.mean(
        axis=1,
        keepdims=True,
    )
    pred_centered = y_pred - y_pred.mean(
        axis=1,
        keepdims=True,
    )

    numerator = np.sum(
        true_centered * pred_centered,
        axis=1,
    )

    denominator = np.sqrt(
        np.sum(true_centered**2, axis=1)
        * np.sum(pred_centered**2, axis=1)
    )

    valid_rows = denominator > epsilon

    if not np.any(valid_rows):
        raise ValueError(
            "No cells have sufficient variance for correlation."
        )

    correlations = np.full(
        y_true.shape[0],
        np.nan,
        dtype=np.float64,
    )
    correlations[valid_rows] = (
        numerator[valid_rows]
        / denominator[valid_rows]
    )

    return float(np.nanmean(correlations)), correlations


def proteinwise_pearson(
    y_true,
    y_pred,
    protein_names=None,
    epsilon=1e-12,
):
    """
    Compute Pearson correlation across cells for each protein.

    This is useful for biological interpretation but differs from
    the Kaggle row-wise metric.
    """
    y_true, y_pred = validate_prediction_arrays(
        y_true,
        y_pred,
    )

    n_proteins = y_true.shape[1]

    if protein_names is None:
        protein_names = [
            f"protein_{index}"
            for index in range(n_proteins)
        ]

    protein_names = np.asarray(
        protein_names,
        dtype=str,
    )

    if len(protein_names) != n_proteins:
        raise ValueError(
            "The number of protein names does not match "
            "the target matrix."
        )

    true_centered = y_true - y_true.mean(
        axis=0,
        keepdims=True,
    )
    pred_centered = y_pred - y_pred.mean(
        axis=0,
        keepdims=True,
    )

    numerator = np.sum(
        true_centered * pred_centered,
        axis=0,
    )

    denominator = np.sqrt(
        np.sum(true_centered**2, axis=0)
        * np.sum(pred_centered**2, axis=0)
    )

    correlations = np.full(
        n_proteins,
        np.nan,
        dtype=np.float64,
    )

    valid_proteins = denominator > epsilon
    correlations[valid_proteins] = (
        numerator[valid_proteins]
        / denominator[valid_proteins]
    )

    result = pd.DataFrame(
        {
            "protein": protein_names,
            "PCC": correlations,
        }
    )

    return result


def root_mean_squared_error(y_true, y_pred):
    """Compute RMSE across all cells and proteins."""
    y_true, y_pred = validate_prediction_arrays(
        y_true,
        y_pred,
    )

    return float(
        np.sqrt(np.mean((y_true - y_pred) ** 2))
    )


def summarize_predictions(
    y_true,
    y_pred,
    protein_names=None,
):
    """Return global and protein-wise evaluation results."""
    kaggle_score, cell_correlations = rowwise_pearson(
        y_true,
        y_pred,
    )

    protein_metrics = proteinwise_pearson(
        y_true,
        y_pred,
        protein_names=protein_names,
    )

    
    
    valid_protein_pcc = protein_metrics["PCC"].to_numpy()
    valid_protein_pcc = valid_protein_pcc[
        np.isfinite(valid_protein_pcc)
    ]

    if len(valid_protein_pcc) > 0:
        mean_proteinwise_pcc = float(
            valid_protein_pcc.mean()
        )
        median_proteinwise_pcc = float(
            np.median(valid_protein_pcc)
        )
    else:
        mean_proteinwise_pcc = None
        median_proteinwise_pcc = None

    summary = {
        "rowwise_pearson": kaggle_score,
        "mean_proteinwise_pcc": mean_proteinwise_pcc,
        "median_proteinwise_pcc": median_proteinwise_pcc,
        "rmse": root_mean_squared_error(
            y_true,
            y_pred,
        ),
        "n_cells": int(y_true.shape[0]),
        "n_proteins": int(y_true.shape[1]),
    }

    return summary, protein_metrics, cell_correlations