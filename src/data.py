from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_csv(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file does not exist: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, received: {path.suffix}")

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError("Loaded dataset is empty.")

    return data


def validate_dataframe(data: pd.DataFrame) -> None:
    if data.empty:
        raise ValueError("Dataset is empty.")

    if data.columns.duplicated().any():
        duplicated = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(f"Duplicated columns detected: {duplicated}")

    numeric_data = data.select_dtypes(include=[np.number])

    if numeric_data.empty:
        raise ValueError("Dataset does not contain numeric columns.")

    if np.isinf(numeric_data.to_numpy()).any():
        raise ValueError("Dataset contains infinite values.")


def chronological_split(
    data: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "The sum of train_ratio and validation_ratio must be lower than 1."
        )

    number_of_rows = len(data)

    train_end = int(number_of_rows * train_ratio)
    validation_end = train_end + int(number_of_rows * validation_ratio)

    train_data = data.iloc[:train_end].copy()
    validation_data = data.iloc[train_end:validation_end].copy()
    test_data = data.iloc[validation_end:].copy()

    return train_data, validation_data, test_data


def save_splits(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    output_directory: str | Path,
) -> None:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    train_data.to_csv(output_path / "train.csv", index=False)
    validation_data.to_csv(output_path / "validation.csv", index=False)
    test_data.to_csv(output_path / "test.csv", index=False)

def prepare_structure(data: pd.DataFrame) -> pd.DataFrame:
    prepared = data.copy()

    prepared = prepared.rename(columns={"Unnamed: 0": "timestamp"})

    prepared.columns = [
        column.strip()
        for column in prepared.columns
    ]

    prepared["timestamp"] = pd.to_datetime(
        prepared["timestamp"],
        errors="raise",
    )

    prepared = prepared.sort_values("timestamp")
    prepared = prepared.drop_duplicates(subset="timestamp")
    prepared = prepared.reset_index(drop=True)

    return prepared