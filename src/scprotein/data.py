"""Data loading, alignment, and splitting utilities for CITE-seq."""

from pathlib import Path

import numpy as np
import pandas as pd


def inspect_hdf_keys(path):
    """Return all DataFrame keys stored in a pandas HDF5 file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with pd.HDFStore(path, mode="r") as store:
        keys = [key.lstrip("/") for key in store.keys()]

    return keys


def load_hdf_dataframe(path, key=None):
    """
    Load a DataFrame from a pandas HDF5 file.

    Parameters
    ----------
    path
        Path to the HDF5 file.
    key
        HDF5 key. When omitted, the file must contain exactly one key.
    """
    path = Path(path)
    available_keys = inspect_hdf_keys(path)

    if key is None:
        if len(available_keys) != 1:
            raise ValueError(
                f"{path} contains {len(available_keys)} keys: "
                f"{available_keys}. Please specify one."
            )
        key = available_keys[0]

    if key not in available_keys:
        raise KeyError(
            f"Key '{key}' was not found in {path}. "
            f"Available keys: {available_keys}"
        )

    print(f"Loading {path.name} with key '{key}'...")
    dataframe = pd.read_hdf(path, key=key)

    if not dataframe.index.is_unique:
        raise ValueError(f"Cell IDs are not unique in {path}")

    print(f"Loaded shape: {dataframe.shape}")

    return dataframe


def load_metadata(path):
    """Load metadata.csv and use cell_id as the row index."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    metadata = pd.read_csv(path)

    if "cell_id" in metadata.columns:
        metadata = metadata.set_index("cell_id")
    else:
        # Handles CSV files in which cell_id was saved as the first column.
        metadata = pd.read_csv(path, index_col=0)
        metadata.index.name = "cell_id"

    required_columns = {
        "day",
        "donor",
        "cell_type",
        "technology",
    }
    missing_columns = required_columns.difference(metadata.columns)

    if missing_columns:
        raise ValueError(
            "Metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if not metadata.index.is_unique:
        raise ValueError("Cell IDs are not unique in metadata.")

    print(f"Metadata shape: {metadata.shape}")
    print("Metadata columns:", metadata.columns.tolist())

    return metadata


def align_cite_data(rna, protein, metadata):
    """
    Align RNA, protein, and metadata by cell_id.

    The original RNA cell order is preserved.
    """
    common_mask = (
        rna.index.isin(protein.index)
        & rna.index.isin(metadata.index)
    )
    common_cell_ids = rna.index[common_mask]

    if len(common_cell_ids) == 0:
        raise ValueError(
            "No common cell IDs were found among RNA, protein, and metadata."
        )

    rna_aligned = rna.loc[common_cell_ids]
    protein_aligned = protein.loc[common_cell_ids]
    metadata_aligned = metadata.loc[common_cell_ids]

    if not rna_aligned.index.equals(protein_aligned.index):
        raise ValueError(
            "RNA and protein cell orders do not match after alignment."
        )

    if not rna_aligned.index.equals(metadata_aligned.index):
        raise ValueError(
            "RNA and metadata cell orders do not match after alignment."
        )

    print("\nAlignment completed")
    print(f"RNA:      {rna_aligned.shape}")
    print(f"Protein:  {protein_aligned.shape}")
    print(f"Metadata: {metadata_aligned.shape}")

    return rna_aligned, protein_aligned, metadata_aligned


def make_day_split(metadata, train_days, validation_days):
    """
    Create a time-based train/validation split.

    Training: Day 2 and Day 3
    Validation: Day 4
    """
    train_days = list(train_days)
    validation_days = list(validation_days)

    train_mask = metadata["day"].isin(
        train_days
    ).to_numpy()

    validation_mask = metadata["day"].isin(
        validation_days
    ).to_numpy()

    if np.any(train_mask & validation_mask):
        raise ValueError(
            "Training and validation splits overlap."
        )

    if train_mask.sum() == 0:
        raise ValueError(
            f"No training cells found for days {train_days}."
        )

    if validation_mask.sum() == 0:
        raise ValueError(
            "No validation cells found for days "
            f"{validation_days}."
        )

    unused_mask = ~(train_mask | validation_mask)

    print("\nTime-based split")
    print(f"Training days:       {train_days}")
    print(f"Validation days:     {validation_days}")
    print(f"Training cells:      {train_mask.sum():,}")
    print(
        f"Validation cells:    "
        f"{validation_mask.sum():,}"
    )
    print(f"Unused cells:        {unused_mask.sum():,}")

    return train_mask, validation_mask

def summarize_metadata(metadata):
    """Create summary tables for days, donors, and cell types."""
    day_counts = (
        metadata["day"]
        .value_counts()
        .sort_index()
        .rename_axis("day")
        .reset_index(name="n_cells")
    )

    donor_counts = (
        metadata["donor"]
        .value_counts()
        .sort_index()
        .rename_axis("donor")
        .reset_index(name="n_cells")
    )

    cell_type_counts = (
        metadata["cell_type"]
        .value_counts()
        .rename_axis("cell_type")
        .reset_index(name="n_cells")
    )

    return {
        "day_counts": day_counts,
        "donor_counts": donor_counts,
        "cell_type_counts": cell_type_counts,
    }