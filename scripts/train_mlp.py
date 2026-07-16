"""Train a leak-free multi-output MLP baseline."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from scprotein.metrics import summarize_predictions
from scprotein.models import ProteinMLP
from scprotein.train import (
    predict_mlp,
    set_random_seed,
    train_mlp,
)


EXPERIMENT_NAME = "EXP02_pca50_multioutput_mlp"


def resolve_path(path_value):
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def main(config_path):
    config_path = resolve_path(config_path)

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    seed = int(config["seed"])
    set_random_seed(seed)

    data_path = (
        resolve_path(
            config["output"]["processed_dir"]
        )
        / "cite_day_split_pca.npz"
    )

    result_directory = (
        resolve_path(config["output"]["result_dir"])
        / EXPERIMENT_NAME
    )
    model_directory = resolve_path(
        config["output"]["model_dir"]
    )

    result_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    with np.load(data_path) as data:
        X_train_full = data["X_train"].astype(
            np.float32
        )
        X_validation = data[
            "X_validation"
        ].astype(np.float32)
        y_train_full = data["y_train"].astype(
            np.float32
        )
        y_validation = data[
            "y_validation"
        ].astype(np.float32)
        validation_cell_ids = data[
            "validation_cell_ids"
        ].astype(str)
        feature_names = data[
            "feature_names"
        ].astype(str)
        protein_names = data[
            "protein_names"
        ].astype(str)

    internal_fraction = float(
        config["mlp"][
            "internal_validation_fraction"
        ]
    )

    all_training_indices = np.arange(
        X_train_full.shape[0]
    )

    fit_indices, internal_indices = (
        train_test_split(
            all_training_indices,
            test_size=internal_fraction,
            random_state=seed,
            shuffle=True,
        )
    )

    # Fit both scalers on the internal training subset only.
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_fit = X_scaler.fit_transform(
        X_train_full[fit_indices]
    ).astype(np.float32)

    X_internal = X_scaler.transform(
        X_train_full[internal_indices]
    ).astype(np.float32)

    X_validation_scaled = X_scaler.transform(
        X_validation
    ).astype(np.float32)

    y_fit = y_scaler.fit_transform(
        y_train_full[fit_indices]
    ).astype(np.float32)

    y_internal = y_scaler.transform(
        y_train_full[internal_indices]
    ).astype(np.float32)

    batch_size = int(
        config["mlp"]["batch_size"]
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_fit),
            torch.from_numpy(y_fit),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )

    internal_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_internal),
            torch.from_numpy(y_internal),
        ),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(
                X_validation_scaled
            )
        ),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    model = ProteinMLP(
        input_dim=X_fit.shape[1],
        output_dim=y_fit.shape[1],
        hidden_dims=tuple(
            config["mlp"]["hidden_dims"]
        ),
        dropout=float(
            config["mlp"]["dropout"]
        ),
    )

    print(model)

    start_time = time.perf_counter()

    (
        model,
        history,
        best_epoch,
        best_internal_loss,
    ) = train_mlp(
        model=model,
        train_loader=train_loader,
        internal_validation_loader=(
            internal_loader
        ),
        learning_rate=float(
            config["mlp"]["learning_rate"]
        ),
        weight_decay=float(
            config["mlp"]["weight_decay"]
        ),
        epochs=int(config["mlp"]["epochs"]),
        patience=int(
            config["mlp"]["patience"]
        ),
        device=device,
    )

    scaled_predictions = predict_mlp(
        model=model,
        data_loader=validation_loader,
        device=device,
    )

    validation_predictions = (
        y_scaler.inverse_transform(
            scaled_predictions
        ).astype(np.float32)
    )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    (
        summary,
        protein_metrics,
        cell_correlations,
    ) = summarize_predictions(
        y_true=y_validation,
        y_pred=validation_predictions,
        protein_names=protein_names,
    )

    summary.update(
        {
            "experiment": EXPERIMENT_NAME,
            "model": "Multi-output MLP",
            "input_dim": int(X_fit.shape[1]),
            "output_dim": int(y_fit.shape[1]),
            "hidden_dims": config["mlp"][
                "hidden_dims"
            ],
            "dropout": float(
                config["mlp"]["dropout"]
            ),
            "best_epoch": int(best_epoch),
            "best_internal_validation_loss": (
                float(best_internal_loss)
            ),
            "training_seconds": float(
                elapsed_seconds
            ),
            "device": str(device),
            "train_days": config["split"][
                "train_days"
            ],
            "validation_days": config["split"][
                "validation_days"
            ],
            "day4_used_for_early_stopping": False,
            "uses_target_derived_features": False,
        }
    )

    history.to_csv(
        result_directory
        / "training_history.csv",
        index=False,
    )

    protein_metrics.to_csv(
        result_directory
        / "per_protein_metrics.csv",
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

    np.save(
        result_directory
        / "validation_predictions.npy",
        validation_predictions,
    )

    with (
        result_directory
        / "metrics_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    checkpoint = {
        "model_state_dict": (
            model.state_dict()
        ),
        "input_dim": int(X_fit.shape[1]),
        "output_dim": int(y_fit.shape[1]),
        "hidden_dims": config["mlp"][
            "hidden_dims"
        ],
        "dropout": float(
            config["mlp"]["dropout"]
        ),
        "feature_names": feature_names.tolist(),
        "protein_names": protein_names.tolist(),
        "best_epoch": int(best_epoch),
    }

    torch.save(
        checkpoint,
        model_directory
        / "pca50_multioutput_mlp.pth",
    )

    joblib.dump(
        X_scaler,
        model_directory
        / "mlp_X_scaler.joblib",
    )
    joblib.dump(
        y_scaler,
        model_directory
        / "mlp_y_scaler.joblib",
    )

    shutil.copy2(
        config_path,
        result_directory / "config.yaml",
    )

    print("\nDay 4 validation results")
    print(
        "Row-wise Pearson:       "
        f"{summary['rowwise_pearson']:.6f}"
    )
    print(
        "Mean protein-wise PCC:  "
        f"{summary['mean_proteinwise_pcc']:.6f}"
    )
    print(
        "Median protein-wise PCC:"
        f" {summary['median_proteinwise_pcc']:.6f}"
    )
    print(f"RMSE:                   {summary['rmse']:.6f}")
    print(f"Best epoch:             {best_epoch}")
    print(f"Time:                   {elapsed_seconds:.2f}s")
    print(f"Results:                {result_directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
    )
    arguments = parser.parse_args()

    main(arguments.config)