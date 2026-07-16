"""Model definitions for protein-abundance prediction."""

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_ridge_model(alpha=1.0):
    """
    Build a multi-output Ridge-regression pipeline.

    StandardScaler and Ridge are both fitted using training cells only.
    """
    if alpha < 0:
        raise ValueError("Ridge alpha must be non-negative.")

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(
                    with_mean=True,
                    with_std=True,
                ),
            ),
            (
                "ridge",
                Ridge(
                    alpha=float(alpha),
                ),
            ),
        ]
    )

    return model


def fit_ridge_model(model, X_train, y_train):
    """Fit Ridge and return the fitted pipeline."""
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            "Training features and targets contain "
            "different numbers of cells."
        )

    print("\nFitting multi-output Ridge regression")
    print(f"Training features: {X_train.shape}")
    print(f"Training targets:  {y_train.shape}")

    model.fit(X_train, y_train)

    return model


def predict_ridge_model(model, X):
    """Generate protein predictions using a fitted Ridge model."""
    predictions = model.predict(X)

    return predictions

import torch
from torch import nn


class ProteinMLP(nn.Module):
    """Multi-output MLP for predicting all proteins jointly."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=(64, 32),
        dropout=0.2,
    ):
        super().__init__()

        if input_dim <= 0 or output_dim <= 0:
            raise ValueError(
                "Input and output dimensions must be positive."
            )

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim

        layers.append(
            nn.Linear(previous_dim, output_dim)
        )

        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)