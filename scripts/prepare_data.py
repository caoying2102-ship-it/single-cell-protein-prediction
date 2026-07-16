"""Prepare leak-free PCA features for CITE-seq protein prediction."""

import argparse
import gc
from importlib.metadata import metadata
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from scprotein.data import (
    align_cite_data,
    load_hdf_dataframe,
    load_metadata,
    make_day_split,
    summarize_metadata,
)
from scprotein.features import (
    fit_pca_on_training_data,
    save_pca_artifacts,
)


def resolve_project_path(path_value):
    """Resolve relative paths against the project root."""
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_config(config_path):
    """Load the YAML configuration file."""
    config_path = resolve_project_path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    print(f"Configuration loaded: {config_path}")

    return config


def save_metadata_summaries(metadata, output_directory):
    """Save simple summaries used for quality control."""
    summaries = summarize_metadata(metadata)

    for summary_name, summary_table in summaries.items():
        output_path = (
            output_directory / f"{summary_name}.csv"
        )
        summary_table.to_csv(output_path, index=False)

    print(f"Metadata summaries saved to: {output_directory}")


def main(config_path):
    config = load_config(config_path)

    seed = int(config["seed"])
    np.random.seed(seed)

    rna_path = resolve_project_path(
        config["data"]["rna"]
    )
    protein_path = resolve_project_path(
        config["data"]["protein"]
    )
    metadata_path = resolve_project_path(
        config["data"]["metadata"]
    )

    processed_directory = resolve_project_path(
        config["output"]["processed_dir"]
    )
    feature_directory = resolve_project_path(
        config["output"]["feature_dir"]
    )
    log_directory = resolve_project_path(
        config["output"]["log_dir"]
    )

    processed_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    feature_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nInput files")
    print(f"RNA:       {rna_path}")
    print(f"Protein:   {protein_path}")
    print(f"Metadata:  {metadata_path}")

    # ---------------------------------------------------------
    # 1. Load raw data
    # ---------------------------------------------------------
    print("\nStep 1/6: Loading raw data")

    rna = load_hdf_dataframe(
        rna_path,
        key=config["data"]["rna_key"],
    )

    protein = load_hdf_dataframe(
        protein_path,
        key=config["data"]["protein_key"],
    )

    metadata = load_metadata(metadata_path)

    # ---------------------------------------------------------
    # 2. Align all modalities by cell_id
    # ---------------------------------------------------------
    print("\nStep 2/6: Aligning RNA, protein, and metadata")

    rna, protein, metadata = align_cite_data(
        rna,
        protein,
        metadata,
    )

    if rna.shape[0] != protein.shape[0]:
        raise ValueError(
            "RNA and protein contain different numbers of cells."
        )

    if rna.shape[0] != metadata.shape[0]:
        raise ValueError(
            "RNA and metadata contain different numbers of cells."
        )

    rna_gene_names = rna.columns.to_numpy(dtype=str)
    protein_names = protein.columns.to_numpy(dtype=str)

    # ---------------------------------------------------------
    # 3. Save descriptive summaries
    # ---------------------------------------------------------
    print("\nStep 3/6: Saving metadata summaries")

    save_metadata_summaries(
        metadata,
        processed_directory,
    )

    # ---------------------------------------------------------
    # 4. Split by developmental day before fitting PCA
    # ---------------------------------------------------------
    print("\nStep 4/6: Creating time-based split")

    train_mask, validation_mask = make_day_split(
        metadata,
        train_days=config["split"]["train_days"],
        validation_days=config["split"][
            "validation_days"
        ],
    )


    train_cell_ids = metadata.index[
    train_mask
    ].to_numpy(dtype=str)

    validation_cell_ids = metadata.index[
    validation_mask
    ].to_numpy(dtype=str)

    train_metadata = metadata.iloc[train_mask].copy()
    validation_metadata = metadata.iloc[
        validation_mask
    ].copy()

    train_metadata.to_csv(
        processed_directory / "metadata_train.csv"
    )
    validation_metadata.to_csv(
        processed_directory / "metadata_validation.csv"
    )

    # ---------------------------------------------------------
    # 5. Convert the two splits to float32 arrays
    # ---------------------------------------------------------
    print("\nStep 5/6: Creating model arrays")

    X_train_raw = rna.iloc[train_mask].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    X_validation_raw = rna.iloc[
        validation_mask
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    y_train = protein.iloc[train_mask].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    y_validation = protein.iloc[
        validation_mask
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    print(f"Raw training RNA:       {X_train_raw.shape}")
    print(
        f"Raw validation RNA:     {X_validation_raw.shape}"
    )
    print(f"Training targets:       {y_train.shape}")
    print(
        f"Validation targets:     {y_validation.shape}"
    )

    # The full DataFrames are no longer required.
    del rna
    del protein
    gc.collect()

    # ---------------------------------------------------------
    # 6. Fit PCA on training cells only
    # ---------------------------------------------------------
    print("\nStep 6/6: Fitting leak-free PCA")

    (
        X_train,
        X_validation,
        pca,
        feature_names,
    ) = fit_pca_on_training_data(
        X_train=X_train_raw,
        X_validation=X_validation_raw,
        n_components=int(
            config["features"]["n_pca_components"]
        ),
        solver=config["features"]["pca_solver"],
        random_state=seed,
    )

    save_pca_artifacts(
        pca=pca,
        feature_names=feature_names,
        output_directory=feature_directory,
    )

    # PCA output is small, so the raw split matrices can be freed.
    del X_train_raw
    del X_validation_raw
    gc.collect()

    # Save names needed for future test-data transformation.
    (
        feature_directory / "rna_gene_names.txt"
    ).write_text(
        "\n".join(rna_gene_names) + "\n",
        encoding="utf-8",
    )

    (
        feature_directory / "protein_names.txt"
    ).write_text(
        "\n".join(protein_names) + "\n",
        encoding="utf-8",
    )

    # Save compact arrays used by Ridge and MLP.
    output_path = (
        processed_directory / "cite_day_split_pca.npz"
    )

    np.savez_compressed(
        output_path,
        X_train=X_train,
        X_validation=X_validation,
        y_train=y_train,
        y_validation=y_validation,
        train_cell_ids=train_cell_ids,
        validation_cell_ids=validation_cell_ids,
        feature_names=np.asarray(
            feature_names,
            dtype=str,
        ),
        protein_names=protein_names,
    )

    split_summary = pd.DataFrame(
        {
            "split": ["train", "validation"],
            "days": [
                ",".join(
                    map(
                        str,
                        config["split"]["train_days"],
                    )
                ),
                ",".join(
                    map(
                        str,
                        config["split"][
                            "validation_days"
                        ],
                    )
                ),
            ],
            "n_cells": [
                int(train_mask.sum()),
                int(validation_mask.sum()),
            ],
            "n_features": [
                X_train.shape[1],
                X_validation.shape[1],
            ],
            "n_targets": [
                y_train.shape[1],
                y_validation.shape[1],
            ],
        }
    )

    split_summary.to_csv(
        processed_directory / "split_summary.csv",
        index=False,
    )

    print("\nData preparation completed successfully")
    print(f"Prepared arrays: {output_path}")
    print(f"Training features:   {X_train.shape}")
    print(f"Validation features: {X_validation.shape}")
    print(f"Training targets:    {y_train.shape}")
    print(f"Validation targets:  {y_validation.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Prepare leak-free CITE-seq PCA features."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )

    arguments = parser.parse_args()
    main(arguments.config)