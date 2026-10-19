from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.datasets import (
    create_dataloader,
    create_sliding_windows,
)
from src.metrics import (
    calculate_regression_metrics,
    inverse_transform_target,
)
from src.models.gru import GRUForecaster
from src.models.mlp import MLPForecaster
from src.models.transformer import TransformerForecaster
from src.training import collect_predictions


PROCESSED_DATA_DIRECTORY = Path("data/processed")
PREPROCESSING_DIRECTORY = Path("results/preprocessing")

CHECKPOINT_DIRECTORY = Path("results/checkpoints")
METRICS_DIRECTORY = Path("results/metrics")
PREDICTIONS_DIRECTORY = Path("results/predictions")

MAPE_MINIMUM_ACTUAL_VALUE_KW = 1e-8


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the best selected model on the test set."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mlp", "gru", "transformer"],
        help="Model architecture to evaluate.",
    )

    return parser.parse_args()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_processed_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
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
        PREPROCESSING_DIRECTORY / "feature_columns.joblib"
    )

    return (
        train_data,
        validation_data,
        test_data,
        feature_columns,
    )


def create_test_split(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    forecast_horizon: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    full_data = pd.concat(
        [
            train_data,
            validation_data,
            test_data,
        ],
        ignore_index=True,
    )

    X, y, target_timestamps = create_sliding_windows(
        data=full_data,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
    )

    test_start = test_data["timestamp"].iloc[0]

    test_mask = target_timestamps >= test_start

    return (
        X[test_mask],
        y[test_mask],
        target_timestamps[test_mask],
    )


def create_model_from_checkpoint(
    checkpoint: dict[str, object],
) -> nn.Module:
    model_name = str(checkpoint["model_name"]).lower()

    configuration = dict(
        checkpoint["model_configuration"]
    )

    if model_name == "mlp":
        return MLPForecaster(
            sequence_length=int(
                checkpoint["sequence_length"]
            ),
            number_of_features=int(
                checkpoint["number_of_features"]
            ),
            hidden_sizes=tuple(
                configuration["hidden_sizes"]
            ),
            dropout=float(
                configuration["dropout"]
            ),
            output_size=1,
        )

    if model_name == "gru":
        return GRUForecaster(
            number_of_features=int(
                checkpoint["number_of_features"]
            ),
            hidden_size=int(
                configuration["hidden_size"]
            ),
            number_of_layers=int(
                configuration["number_of_layers"]
            ),
            dense_size=int(
                configuration["dense_size"]
            ),
            dropout=float(
                configuration["dropout"]
            ),
            output_size=1,
        )

    if model_name == "transformer":
        return TransformerForecaster(
            sequence_length=int(
                checkpoint["sequence_length"]
            ),
            number_of_features=int(
                checkpoint["number_of_features"]
            ),
            d_model=int(
                configuration["d_model"]
            ),
            number_of_heads=int(
                configuration["number_of_heads"]
            ),
            number_of_layers=int(
                configuration["number_of_layers"]
            ),
            dim_feedforward=int(
                configuration["dim_feedforward"]
            ),
            dropout=float(
                configuration["dropout"]
            ),
            output_size=1,
        )

    raise ValueError(
        f"Unsupported model in checkpoint: {model_name}"
    )


def load_model_from_checkpoint(
    checkpoint: dict[str, object],
    device: torch.device,
) -> nn.Module:
    model = create_model_from_checkpoint(
        checkpoint=checkpoint,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model


def save_metrics(
    model_name: str,
    metrics: dict[str, object],
) -> Path:
    METRICS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        METRICS_DIRECTORY
        / f"best_{model_name}_metrics.csv"
    )

    pd.DataFrame([metrics]).to_csv(
        output_path,
        index=False,
    )

    return output_path


def save_predictions(
    model_name: str,
    timestamps: np.ndarray,
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    y_true_kw: np.ndarray,
    y_pred_kw: np.ndarray,
) -> Path:
    PREDICTIONS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        PREDICTIONS_DIRECTORY
        / f"best_{model_name}_predictions.csv"
    )

    predictions_frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "y_true_scaled": y_true_scaled.reshape(-1),
            "y_pred_scaled": y_pred_scaled.reshape(-1),
            "y_true_kw": y_true_kw.reshape(-1),
            "y_pred_kw": y_pred_kw.reshape(-1),
            "error_kw": (
                y_pred_kw.reshape(-1)
                - y_true_kw.reshape(-1)
            ),
            "absolute_error_kw": np.abs(
                y_pred_kw.reshape(-1)
                - y_true_kw.reshape(-1)
            ),
        }
    )

    predictions_frame.to_csv(
        output_path,
        index=False,
    )

    return output_path


def print_target_diagnostics(
    y_true_kw: np.ndarray,
) -> None:
    targets = y_true_kw.reshape(-1)
    absolute_targets = np.abs(targets)

    print("\nTarget diagnostics:")
    print(f"Number of samples:    {len(targets)}")
    print(
        f"Minimum target:       "
        f"{targets.min():.12f} kW"
    )
    print(
        f"Maximum target:       "
        f"{targets.max():.6f} kW"
    )
    print(
        f"Mean target:          "
        f"{targets.mean():.6f} kW"
    )
    print(
        f"Median target:        "
        f"{np.median(targets):.6f} kW"
    )
    print(
        f"Standard deviation:   "
        f"{targets.std():.6f} kW"
    )

    percentiles = [
        0,
        1,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
        100,
    ]

    percentile_values = np.percentile(
        targets,
        percentiles,
    )

    print("\nTarget percentiles:")

    for percentile, value in zip(
        percentiles,
        percentile_values,
    ):
        print(
            f"{percentile:>3}%: "
            f"{value:.6f} kW"
        )

    ranges = [
        (0.0, 0.001),
        (0.001, 0.1),
        (0.1, 1.0),
        (1.0, 5.0),
        (5.0, 10.0),
        (10.0, 20.0),
        (20.0, float("inf")),
    ]

    print("\nTarget ranges:")

    for lower_bound, upper_bound in ranges:
        if np.isinf(upper_bound):
            mask = absolute_targets > lower_bound
            label = f"> {lower_bound:.3f} kW"
        else:
            mask = (
                (absolute_targets > lower_bound)
                & (absolute_targets <= upper_bound)
            )
            label = (
                f"({lower_bound:.3f}, "
                f"{upper_bound:.3f}] kW"
            )

        count = int(mask.sum())
        percentage = count / len(targets) * 100.0

        print(
            f"{label:<22} "
            f"{count:>5} "
            f"({percentage:>6.2f}%)"
        )


def add_checkpoint_metadata(
    metrics: dict[str, object],
    checkpoint: dict[str, object],
    checkpoint_path: Path,
) -> None:
    metrics["model_name"] = checkpoint["model_name"]
    metrics["model_class"] = checkpoint["model_class"]
    metrics["checkpoint"] = checkpoint_path.name
    metrics["trial_number"] = checkpoint["trial_number"]
    metrics["configuration_type"] = (
        checkpoint["configuration_type"]
    )
    metrics["best_epoch"] = checkpoint["epoch"]
    metrics["validation_loss"] = (
        checkpoint["validation_loss"]
    )
    metrics["validation_rmse"] = checkpoint.get(
        "validation_rmse",
        float(
            np.sqrt(
                float(checkpoint["validation_loss"])
            )
        ),
    )
    metrics["sequence_length"] = (
        checkpoint["sequence_length"]
    )
    metrics["forecast_horizon"] = (
        checkpoint["forecast_horizon"]
    )
    metrics["batch_size"] = checkpoint["batch_size"]
    metrics["learning_rate"] = (
        checkpoint["learning_rate"]
    )
    metrics["random_seed"] = checkpoint["random_seed"]
    metrics["mape_minimum_actual_value_kw"] = (
        MAPE_MINIMUM_ACTUAL_VALUE_KW
    )

    configuration = dict(
        checkpoint["model_configuration"]
    )

    for key, value in configuration.items():
        if isinstance(value, tuple):
            metrics[key] = str(value)
        else:
            metrics[key] = value


def main() -> None:
    arguments = parse_arguments()

    requested_model_name = arguments.model.lower()

    checkpoint_path = (
        CHECKPOINT_DIRECTORY
        / f"best_{requested_model_name}.pt"
    )

    device = select_device()

    print("Evaluation configuration:")
    print(f"Model:        {requested_model_name}")
    print(f"Device:       {device}")
    print(f"Checkpoint:   {checkpoint_path}")
    print(
        f"MAPE threshold: "
        f"{MAPE_MINIMUM_ACTUAL_VALUE_KW:.10f} kW"
    )

    if device.type == "cuda":
        print(
            f"GPU:          "
            f"{torch.cuda.get_device_name(0)}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    checkpoint_model_name = str(
        checkpoint["model_name"]
    ).lower()

    if checkpoint_model_name != requested_model_name:
        raise ValueError(
            "Checkpoint model mismatch. "
            f"Requested {requested_model_name}, "
            f"but checkpoint contains "
            f"{checkpoint_model_name}."
        )

    (
        train_data,
        validation_data,
        test_data,
        feature_columns,
    ) = load_processed_data()

    (
        X_test,
        y_test,
        test_timestamps,
    ) = create_test_split(
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        feature_columns=feature_columns,
        sequence_length=int(
            checkpoint["sequence_length"]
        ),
        forecast_horizon=int(
            checkpoint["forecast_horizon"]
        ),
    )

    test_loader = create_dataloader(
        X=X_test,
        y=y_test,
        batch_size=int(
            checkpoint["batch_size"]
        ),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = load_model_from_checkpoint(
        checkpoint=checkpoint,
        device=device,
    )

    (
        targets_tensor,
        predictions_tensor,
    ) = collect_predictions(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    y_true_scaled = targets_tensor.numpy()
    y_pred_scaled = predictions_tensor.numpy()

    scaler = joblib.load(
        PREPROCESSING_DIRECTORY / "scaler.joblib"
    )

    y_true_kw = inverse_transform_target(
        values=y_true_scaled,
        scaler=scaler,
        feature_columns=feature_columns,
    )

    y_pred_kw = inverse_transform_target(
        values=y_pred_scaled,
        scaler=scaler,
        feature_columns=feature_columns,
    )

    print_target_diagnostics(
        y_true_kw=y_true_kw,
    )

    metrics = calculate_regression_metrics(
        y_true=y_true_kw,
        y_pred=y_pred_kw,
        mape_minimum_actual_value=(
            MAPE_MINIMUM_ACTUAL_VALUE_KW
        ),
    )

    add_checkpoint_metadata(
        metrics=metrics,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
    )

    metrics_path = save_metrics(
        model_name=requested_model_name,
        metrics=metrics,
    )

    predictions_path = save_predictions(
        model_name=requested_model_name,
        timestamps=test_timestamps,
        y_true_scaled=y_true_scaled,
        y_pred_scaled=y_pred_scaled,
        y_true_kw=y_true_kw,
        y_pred_kw=y_pred_kw,
    )

    print("\nTest dataset:")
    print(f"X_test:       {X_test.shape}")
    print(f"y_test:       {y_test.shape}")

    print("\nTest metrics:")
    print(
        f"MSE:                  "
        f"{metrics['mse']:.6f}"
    )
    print(
        f"RMSE:                 "
        f"{metrics['rmse']:.6f} kW"
    )
    print(
        f"MAE:                  "
        f"{metrics['mae']:.6f} kW"
    )
    print(
        f"MAPE:                 "
        f"{metrics['mape']:.6f}%"
    )
    print(
        f"R2:                   "
        f"{metrics['r2']:.6f}"
    )
    print(
        f"Explained variance:   "
        f"{metrics['explained_variance']:.6f}"
    )
    print(
        f"Samples:              "
        f"{metrics['number_of_samples']}"
    )
    print(
        f"MAPE valid samples:   "
        f"{metrics['mape_valid_samples']}"
    )
    print(
        f"MAPE excluded:        "
        f"{metrics['mape_excluded_samples']}"
    )
    print(
        f"MAPE threshold:       "
        f"{MAPE_MINIMUM_ACTUAL_VALUE_KW:.10f} kW"
    )

    print("\nSaved files:")
    print(f"Metrics:      {metrics_path}")
    print(f"Predictions:  {predictions_path}")


if __name__ == "__main__":
    main()