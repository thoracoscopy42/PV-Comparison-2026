from pathlib import Path

import numpy as np
import pandas as pd


TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "Active Power (kW)"


def load_csv(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    data = pd.read_csv(file_path)

    if data.empty:
        raise ValueError("Dataset is empty.")

    return data


def prepare_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data.columns = [column.strip() for column in data.columns]

    if "Unnamed: 0" in data.columns:
        data = data.rename(columns={"Unnamed: 0": TIMESTAMP_COLUMN})

    if TIMESTAMP_COLUMN not in data.columns:
        raise ValueError(f"Missing column: {TIMESTAMP_COLUMN}")

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    data[TIMESTAMP_COLUMN] = pd.to_datetime(
        data[TIMESTAMP_COLUMN],
        errors="raise",
    )

    data = data.sort_values(TIMESTAMP_COLUMN)
    data = data.drop_duplicates(subset=TIMESTAMP_COLUMN)
    data = data.reset_index(drop=True)

    return data


def validate_time_index(
    data: pd.DataFrame,
    expected_frequency: str = "15min",
) -> None:
    timestamps = data[TIMESTAMP_COLUMN]

    if not timestamps.is_monotonic_increasing:
        raise ValueError("Timestamps are not sorted chronologically.")

    expected_timestamps = pd.date_range(
        start=timestamps.iloc[0],
        end=timestamps.iloc[-1],
        freq=expected_frequency,
    )

    missing_timestamps = expected_timestamps.difference(
        pd.DatetimeIndex(timestamps)
    )

    duplicated_timestamps = timestamps.duplicated().sum()

    print("\nTime index:")
    print(f"First timestamp:       {timestamps.iloc[0]}")
    print(f"Last timestamp:        {timestamps.iloc[-1]}")
    print(f"Expected timestamps:   {len(expected_timestamps)}")
    print(f"Actual timestamps:     {len(timestamps)}")
    print(f"Missing timestamps:    {len(missing_timestamps)}")
    print(f"Duplicated timestamps: {duplicated_timestamps}")

    if len(missing_timestamps) > 0:
        raise ValueError("The time index contains missing timestamps.")


def replace_infinite_values(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return data


def chronological_split(
    data: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_ratio <= 0 or validation_ratio <= 0:
        raise ValueError("Split ratios must be greater than zero.")

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "Train and validation ratios must sum to less than one."
        )

    number_of_rows = len(data)

    train_end = int(number_of_rows * train_ratio)
    validation_end = int(
        number_of_rows * (train_ratio + validation_ratio)
    )

    train_data = data.iloc[:train_end].copy()
    validation_data = data.iloc[train_end:validation_end].copy()
    test_data = data.iloc[validation_end:].copy()

    return train_data, validation_data, test_data


def print_missing_values(
    name: str,
    data: pd.DataFrame,
) -> None:
    missing = data.isna().sum()
    missing = missing[missing > 0]

    print(f"\nMissing values — {name}:")

    if missing.empty:
        print("None")
    else:
        print(missing.to_string())


def save_splits(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    output_directory: str | Path,
) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    train_data.to_csv(output_directory / "train.csv", index=False)
    validation_data.to_csv(
        output_directory / "validation.csv",
        index=False,
    )
    test_data.to_csv(output_directory / "test.csv", index=False)