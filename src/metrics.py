from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.data import TARGET_COLUMN


def prepare_regression_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    true_values = np.asarray(
        y_true,
        dtype=np.float64,
    ).reshape(-1)

    predicted_values = np.asarray(
        y_pred,
        dtype=np.float64,
    ).reshape(-1)

    if true_values.size == 0:
        raise ValueError(
            "y_true and y_pred cannot be empty."
        )

    if true_values.shape != predicted_values.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape. "
            f"Received {true_values.shape} and "
            f"{predicted_values.shape}."
        )

    if not np.isfinite(true_values).all():
        raise ValueError(
            "y_true contains NaN or infinite values."
        )

    if not np.isfinite(predicted_values).all():
        raise ValueError(
            "y_pred contains NaN or infinite values."
        )

    return true_values, predicted_values


def inverse_transform_target(
    values: np.ndarray,
    scaler: Any,
    feature_columns: list[str],
    target_column: str = TARGET_COLUMN,
) -> np.ndarray:
    transformed_values = np.asarray(
        values,
        dtype=np.float64,
    )

    if transformed_values.size == 0:
        raise ValueError(
            "values cannot be empty."
        )

    if not np.isfinite(transformed_values).all():
        raise ValueError(
            "values contain NaN or infinite values."
        )

    if target_column not in feature_columns:
        raise ValueError(
            f"Target column '{target_column}' is not present "
            "in feature_columns."
        )

    target_index = feature_columns.index(
        target_column
    )

    if not hasattr(scaler, "mean_"):
        raise ValueError(
            "The scaler does not contain fitted mean values."
        )

    if not hasattr(scaler, "scale_"):
        raise ValueError(
            "The scaler does not contain fitted scale values."
        )

    if target_index >= len(scaler.mean_):
        raise ValueError(
            "The scaler dimensions do not match feature_columns."
        )

    if target_index >= len(scaler.scale_):
        raise ValueError(
            "The scaler dimensions do not match feature_columns."
        )

    target_mean = float(
        scaler.mean_[target_index]
    )

    target_scale = float(
        scaler.scale_[target_index]
    )

    original_values = (
        transformed_values * target_scale
        + target_mean
    )

    return original_values.astype(
        np.float64,
        copy=False,
    )


def calculate_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    minimum_actual_value: float = 1e-8,
) -> tuple[float, int, int]:
    if minimum_actual_value < 0:
        raise ValueError(
            "minimum_actual_value cannot be negative."
        )

    true_values, predicted_values = (
        prepare_regression_arrays(
            y_true=y_true,
            y_pred=y_pred,
        )
    )

    valid_mask = (
        np.abs(true_values)
        > minimum_actual_value
    )

    valid_samples = int(valid_mask.sum())

    excluded_samples = int(
        len(true_values) - valid_samples
    )

    if valid_samples == 0:
        return float("nan"), valid_samples, excluded_samples

    percentage_errors = np.abs(
        (
            true_values[valid_mask]
            - predicted_values[valid_mask]
        )
        / true_values[valid_mask]
    )

    mape = float(
        np.mean(percentage_errors) * 100.0
    )

    return mape, valid_samples, excluded_samples


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mape_minimum_actual_value: float = 1e-8,
) -> dict[str, float | int]:
    true_values, predicted_values = (
        prepare_regression_arrays(
            y_true=y_true,
            y_pred=y_pred,
        )
    )

    mse = float(
        mean_squared_error(
            true_values,
            predicted_values,
        )
    )

    rmse = float(
        np.sqrt(mse)
    )

    mae = float(
        mean_absolute_error(
            true_values,
            predicted_values,
        )
    )

    mape, mape_valid_samples, mape_excluded_samples = (
        calculate_mape(
            y_true=true_values,
            y_pred=predicted_values,
            minimum_actual_value=(
                mape_minimum_actual_value
            ),
        )
    )

    r2 = float(
        r2_score(
            true_values,
            predicted_values,
        )
    )

    explained_variance = float(
        explained_variance_score(
            true_values,
            predicted_values,
        )
    )

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "explained_variance": explained_variance,
        "number_of_samples": int(
            len(true_values)
        ),
        "mape_valid_samples": mape_valid_samples,
        "mape_excluded_samples": (
            mape_excluded_samples
        ),
    }