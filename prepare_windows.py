from pathlib import Path

import joblib
import pandas as pd

from src.datasets import (
    create_dataloader,
    create_sliding_windows,
)


PROCESSED_DATA_DIRECTORY = Path(
    "data/processed"
)

PREPROCESSING_DIRECTORY = Path(
    "results/preprocessing"
)

SEQUENCE_LENGTH = 32
FORECAST_HORIZON = 1
BATCH_SIZE = 256


def main() -> None:
    train_data = pd.read_csv(
        PROCESSED_DATA_DIRECTORY / "train.csv",
        parse_dates=["timestamp"],
    )

    validation_data = pd.read_csv(
        PROCESSED_DATA_DIRECTORY / "validation.csv",
        parse_dates=["timestamp"],
    )

    test_data = pd.read_csv(
        PROCESSED_DATA_DIRECTORY / "test.csv",
        parse_dates=["timestamp"],
    )

    feature_columns = joblib.load(
        PREPROCESSING_DIRECTORY
        / "feature_columns.joblib"
    )

    full_data = pd.concat(
        [
            train_data,
            validation_data,
            test_data,
        ],
        ignore_index=True,
    )

    X, y, target_timestamps = (
        create_sliding_windows(
            data=full_data,
            feature_columns=feature_columns,
            sequence_length=SEQUENCE_LENGTH,
            forecast_horizon=FORECAST_HORIZON,
        )
    )

    validation_start = (
        validation_data["timestamp"].iloc[0]
    )

    test_start = (
        test_data["timestamp"].iloc[0]
    )

    train_mask = (
        target_timestamps < validation_start
    )

    validation_mask = (
        (target_timestamps >= validation_start)
        & (target_timestamps < test_start)
    )

    test_mask = (
        target_timestamps >= test_start
    )

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_validation = X[validation_mask]
    y_validation = y[validation_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    train_loader = create_dataloader(
        X=X_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    validation_loader = create_dataloader(
        X=X_validation,
        y=y_validation,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = create_dataloader(
        X=X_test,
        y=y_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print("Sliding-window configuration:")
    print(
        f"Sequence length:    "
        f"{SEQUENCE_LENGTH}"
    )
    print(
        f"Forecast horizon:   "
        f"{FORECAST_HORIZON}"
    )
    print(
        f"Number of features: "
        f"{X.shape[2]}"
    )

    print("\nShapes:")
    print(
        f"X_train:       "
        f"{X_train.shape}"
    )
    print(
        f"y_train:       "
        f"{y_train.shape}"
    )
    print(
        f"X_validation:  "
        f"{X_validation.shape}"
    )
    print(
        f"y_validation:  "
        f"{y_validation.shape}"
    )
    print(
        f"X_test:        "
        f"{X_test.shape}"
    )
    print(
        f"y_test:        "
        f"{y_test.shape}"
    )

    print("\nFirst sample:")
    print(
        f"Input window shape: "
        f"{X_train[0].shape}"
    )
    print(
        f"Target shape:       "
        f"{y_train[0].shape}"
    )
    print(
        f"Target value:       "
        f"{y_train[0, 0]:.6f}"
    )

    first_batch_X, first_batch_y = next(
        iter(train_loader)
    )

    print("\nDataLoader check:")
    print(
        f"Batch size:          "
        f"{BATCH_SIZE}"
    )
    print(
        f"Train batches:       "
        f"{len(train_loader)}"
    )
    print(
        f"Validation batches:  "
        f"{len(validation_loader)}"
    )
    print(
        f"Test batches:        "
        f"{len(test_loader)}"
    )
    print(
        f"First batch X shape: "
        f"{tuple(first_batch_X.shape)}"
    )
    print(
        f"First batch y shape: "
        f"{tuple(first_batch_y.shape)}"
    )
    print(
        f"X dtype:             "
        f"{first_batch_X.dtype}"
    )
    print(
        f"y dtype:             "
        f"{first_batch_y.dtype}"
    )


if __name__ == "__main__":
    main()