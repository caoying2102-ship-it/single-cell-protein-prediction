"""PyTorch training utilities."""

import random

import numpy as np
import pandas as pd
import torch
from torch import nn


def set_random_seed(seed):
    """Set reproducible random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_mlp(
    model,
    train_loader,
    internal_validation_loader,
    learning_rate,
    weight_decay,
    epochs,
    patience,
    device,
):
    """Train an MLP using an internal validation split."""
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()

    best_validation_loss = float("inf")
    best_state = None
    best_epoch = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()

        training_loss_sum = 0.0
        training_sample_count = 0

        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(
                device,
                non_blocking=True,
            )
            batch_targets = batch_targets.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad()

            predictions = model(batch_features)
            loss = criterion(
                predictions,
                batch_targets,
            )

            loss.backward()
            optimizer.step()

            batch_size = batch_features.shape[0]
            training_loss_sum += (
                loss.item() * batch_size
            )
            training_sample_count += batch_size

        training_loss = (
            training_loss_sum
            / training_sample_count
        )

        model.eval()

        validation_loss_sum = 0.0
        validation_sample_count = 0

        with torch.no_grad():
            for (
                batch_features,
                batch_targets,
            ) in internal_validation_loader:
                batch_features = batch_features.to(
                    device,
                    non_blocking=True,
                )
                batch_targets = batch_targets.to(
                    device,
                    non_blocking=True,
                )

                predictions = model(batch_features)
                loss = criterion(
                    predictions,
                    batch_targets,
                )

                batch_size = batch_features.shape[0]
                validation_loss_sum += (
                    loss.item() * batch_size
                )
                validation_sample_count += batch_size

        validation_loss = (
            validation_loss_sum
            / validation_sample_count
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": training_loss,
                "internal_validation_loss": (
                    validation_loss
                ),
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train={training_loss:.6f} | "
            f"internal_val={validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter
                in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                "Early stopping at epoch "
                f"{epoch}; best epoch was {best_epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError(
            "Training finished without a valid checkpoint."
        )

    model.load_state_dict(best_state)

    history_table = pd.DataFrame(history)

    return (
        model,
        history_table,
        best_epoch,
        best_validation_loss,
    )


def predict_mlp(model, data_loader, device):
    """Generate batched predictions."""
    model = model.to(device)
    model.eval()

    output_batches = []

    with torch.no_grad():
        for batch in data_loader:
            batch_features = batch[0].to(
                device,
                non_blocking=True,
            )
            predictions = model(batch_features)
            output_batches.append(
                predictions.cpu().numpy()
            )

    return np.concatenate(
        output_batches,
        axis=0,
    )