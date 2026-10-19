from __future__ import annotations

import argparse
import random
import shutil
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from configs.gru import (
    GRU_BASELINE_CONFIG,
    GRU_SEARCH_SPACE,
)
from configs.mlp import (
    MLP_BASELINE_CONFIG,
    MLP_SEARCH_SPACE,
)
from configs.transformer import (
    TRANSFORMER_BASELINE_CONFIG,
    TRANSFORMER_SEARCH_SPACE,
)
from src.datasets import (
    create_dataloader,
    create_sliding_windows,
)
from src.models.gru import GRUForecaster
from src.models.mlp import MLPForecaster
from src.models.transformer import TransformerForecaster
from src.training import (
    evaluate_one_epoch,
    train_one_epoch,
)


PROCESSED_DATA_DIRECTORY = Path("data/processed")
PREPROCESSING_DIRECTORY = Path("results/preprocessing")

CHECKPOINT_DIRECTORY = Path("results/checkpoints")
TRIAL_CHECKPOINT_DIRECTORY = CHECKPOINT_DIRECTORY / "trials"

HISTORY_DIRECTORY = Path("results/history")
SEARCH_DIRECTORY = Path("results/search")

SEQUENCE_LENGTH = 32
FORECAST_HORIZON = 1

MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 15
MINIMUM_IMPROVEMENT = 1e-5

RANDOM_SEED = 42


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one baseline configuration and a selected "
            "number of random configurations."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mlp", "gru", "transformer"],
        help="Model architecture to train.",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help=(
            "Number of random trials. "
            "The baseline trial is added automatically."
        ),
    )

    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def create_data_splits(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
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
        sequence_length=SEQUENCE_LENGTH,
        forecast_horizon=FORECAST_HORIZON,
    )

    validation_start = validation_data["timestamp"].iloc[0]
    test_start = test_data["timestamp"].iloc[0]

    train_mask = target_timestamps < validation_start

    validation_mask = (
        (target_timestamps >= validation_start)
        & (target_timestamps < test_start)
    )

    test_mask = target_timestamps >= test_start

    return (
        X[train_mask],
        y[train_mask],
        X[validation_mask],
        y[validation_mask],
        X[test_mask],
        y[test_mask],
    )


def get_baseline_configuration(
    model_name: str,
) -> dict[str, object]:
    if model_name == "mlp":
        return dict(MLP_BASELINE_CONFIG)

    if model_name == "gru":
        return dict(GRU_BASELINE_CONFIG)

    if model_name == "transformer":
        return dict(TRANSFORMER_BASELINE_CONFIG)

    raise ValueError(f"Unsupported model: {model_name}")


def get_search_space(
    model_name: str,
) -> dict[str, list]:
    if model_name == "mlp":
        return MLP_SEARCH_SPACE

    if model_name == "gru":
        return GRU_SEARCH_SPACE

    if model_name == "transformer":
        return TRANSFORMER_SEARCH_SPACE

    raise ValueError(f"Unsupported model: {model_name}")


def configuration_to_tuple(
    model_name: str,
    configuration: dict[str, object],
) -> tuple[object, ...]:
    if model_name == "mlp":
        return (
            tuple(configuration["hidden_sizes"]),
            float(configuration["dropout"]),
            int(configuration["batch_size"]),
            float(configuration["learning_rate"]),
        )

    if model_name == "gru":
        return (
            int(configuration["hidden_size"]),
            int(configuration["number_of_layers"]),
            int(configuration["dense_size"]),
            float(configuration["dropout"]),
            int(configuration["batch_size"]),
            float(configuration["learning_rate"]),
            float(configuration["gradient_clip_norm"]),
        )

    if model_name == "transformer":
        return (
            int(configuration["d_model"]),
            int(configuration["number_of_heads"]),
            int(configuration["number_of_layers"]),
            int(configuration["dim_feedforward"]),
            float(configuration["dropout"]),
            int(configuration["batch_size"]),
            float(configuration["learning_rate"]),
            float(configuration["gradient_clip_norm"]),
        )

    raise ValueError(f"Unsupported model: {model_name}")


def build_all_configurations(
    model_name: str,
) -> list[dict[str, object]]:
    search_space = get_search_space(model_name)

    if model_name == "mlp":
        return [
            {
                "hidden_sizes": tuple(hidden_sizes),
                "dropout": float(dropout),
                "batch_size": int(batch_size),
                "learning_rate": float(learning_rate),
            }
            for (
                hidden_sizes,
                dropout,
                batch_size,
                learning_rate,
            ) in product(
                search_space["hidden_sizes"],
                search_space["dropout"],
                search_space["batch_size"],
                search_space["learning_rate"],
            )
        ]

    if model_name == "gru":
        return [
            {
                "hidden_size": int(hidden_size),
                "number_of_layers": int(number_of_layers),
                "dense_size": int(dense_size),
                "dropout": float(dropout),
                "batch_size": int(batch_size),
                "learning_rate": float(learning_rate),
                "gradient_clip_norm": float(gradient_clip_norm),
            }
            for (
                hidden_size,
                number_of_layers,
                dense_size,
                dropout,
                batch_size,
                learning_rate,
                gradient_clip_norm,
            ) in product(
                search_space["hidden_size"],
                search_space["number_of_layers"],
                search_space["dense_size"],
                search_space["dropout"],
                search_space["batch_size"],
                search_space["learning_rate"],
                search_space["gradient_clip_norm"],
            )
        ]

    if model_name == "transformer":
        configurations: list[dict[str, object]] = []

        for (
            d_model,
            number_of_heads,
            number_of_layers,
            dim_feedforward,
            dropout,
            batch_size,
            learning_rate,
            gradient_clip_norm,
        ) in product(
            search_space["d_model"],
            search_space["number_of_heads"],
            search_space["number_of_layers"],
            search_space["dim_feedforward"],
            search_space["dropout"],
            search_space["batch_size"],
            search_space["learning_rate"],
            search_space["gradient_clip_norm"],
        ):
            if int(d_model) % int(number_of_heads) != 0:
                continue

            configurations.append(
                {
                    "d_model": int(d_model),
                    "number_of_heads": int(number_of_heads),
                    "number_of_layers": int(number_of_layers),
                    "dim_feedforward": int(dim_feedforward),
                    "dropout": float(dropout),
                    "batch_size": int(batch_size),
                    "learning_rate": float(learning_rate),
                    "gradient_clip_norm": float(
                        gradient_clip_norm
                    ),
                }
            )

        return configurations

    raise ValueError(f"Unsupported model: {model_name}")


def sample_random_configurations(
    model_name: str,
    number_of_trials: int,
    random_seed: int,
) -> list[dict[str, object]]:
    if number_of_trials < 0:
        raise ValueError(
            "number_of_trials cannot be negative."
        )

    all_configurations = build_all_configurations(
        model_name=model_name
    )

    baseline_configuration = get_baseline_configuration(
        model_name=model_name
    )

    baseline_tuple = configuration_to_tuple(
        model_name=model_name,
        configuration=baseline_configuration,
    )

    random_candidates = [
        configuration
        for configuration in all_configurations
        if configuration_to_tuple(
            model_name=model_name,
            configuration=configuration,
        )
        != baseline_tuple
    ]

    if number_of_trials > len(random_candidates):
        raise ValueError(
            f"Requested {number_of_trials} random trials, "
            f"but only {len(random_candidates)} unique "
            "non-baseline configurations are available."
        )

    random_generator = random.Random(random_seed)

    return random_generator.sample(
        random_candidates,
        k=number_of_trials,
    )


def create_model(
    model_name: str,
    number_of_features: int,
    configuration: dict[str, object],
) -> nn.Module:
    if model_name == "mlp":
        return MLPForecaster(
            sequence_length=SEQUENCE_LENGTH,
            number_of_features=number_of_features,
            hidden_sizes=tuple(
                configuration["hidden_sizes"]
            ),
            dropout=float(configuration["dropout"]),
            output_size=1,
        )

    if model_name == "gru":
        return GRUForecaster(
            number_of_features=number_of_features,
            hidden_size=int(configuration["hidden_size"]),
            number_of_layers=int(
                configuration["number_of_layers"]
            ),
            dense_size=int(configuration["dense_size"]),
            dropout=float(configuration["dropout"]),
            output_size=1,
        )

    if model_name == "transformer":
        return TransformerForecaster(
            sequence_length=SEQUENCE_LENGTH,
            number_of_features=number_of_features,
            d_model=int(configuration["d_model"]),
            number_of_heads=int(
                configuration["number_of_heads"]
            ),
            number_of_layers=int(
                configuration["number_of_layers"]
            ),
            dim_feedforward=int(
                configuration["dim_feedforward"]
            ),
            dropout=float(configuration["dropout"]),
            output_size=1,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def get_gradient_clip_norm(
    model_name: str,
    configuration: dict[str, object],
) -> float | None:
    if model_name in {"gru", "transformer"}:
        return float(configuration["gradient_clip_norm"])

    return None


def get_model_configuration(
    model_name: str,
    configuration: dict[str, object],
) -> dict[str, object]:
    if model_name == "mlp":
        return {
            "hidden_sizes": tuple(
                configuration["hidden_sizes"]
            ),
            "dropout": float(configuration["dropout"]),
        }

    if model_name == "gru":
        return {
            "hidden_size": int(configuration["hidden_size"]),
            "number_of_layers": int(
                configuration["number_of_layers"]
            ),
            "dense_size": int(configuration["dense_size"]),
            "dropout": float(configuration["dropout"]),
            "gradient_clip_norm": float(
                configuration["gradient_clip_norm"]
            ),
        }

    if model_name == "transformer":
        return {
            "d_model": int(configuration["d_model"]),
            "number_of_heads": int(
                configuration["number_of_heads"]
            ),
            "number_of_layers": int(
                configuration["number_of_layers"]
            ),
            "dim_feedforward": int(
                configuration["dim_feedforward"]
            ),
            "dropout": float(configuration["dropout"]),
            "gradient_clip_norm": float(
                configuration["gradient_clip_norm"]
            ),
        }

    raise ValueError(f"Unsupported model: {model_name}")


def save_trial_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_name: str,
    trial_number: int,
    configuration_type: str,
    configuration: dict[str, object],
    epoch: int,
    train_loss: float,
    validation_loss: float,
    number_of_features: int,
    trial_seed: int,
) -> Path:
    TRIAL_CHECKPOINT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "_baseline"
        if configuration_type == "baseline"
        else ""
    )

    checkpoint_path = (
        TRIAL_CHECKPOINT_DIRECTORY
        / (
            f"{model_name}_trial_"
            f"{trial_number:03d}"
            f"{suffix}.pt"
        )
    )

    checkpoint = {
        "model_name": model_name,
        "model_class": model.__class__.__name__,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "trial_number": trial_number,
        "configuration_type": configuration_type,
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "validation_rmse": float(
            np.sqrt(validation_loss)
        ),
        "sequence_length": SEQUENCE_LENGTH,
        "forecast_horizon": FORECAST_HORIZON,
        "number_of_features": number_of_features,
        "batch_size": int(configuration["batch_size"]),
        "learning_rate": float(
            configuration["learning_rate"]
        ),
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": (
            EARLY_STOPPING_PATIENCE
        ),
        "minimum_improvement": MINIMUM_IMPROVEMENT,
        "random_seed": trial_seed,
        "model_configuration": get_model_configuration(
            model_name=model_name,
            configuration=configuration,
        ),
    }

    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def save_trial_history(
    model_name: str,
    trial_number: int,
    configuration_type: str,
    history: list[dict[str, float | int]],
) -> Path:
    HISTORY_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "_baseline"
        if configuration_type == "baseline"
        else ""
    )

    history_path = (
        HISTORY_DIRECTORY
        / (
            f"{model_name}_trial_"
            f"{trial_number:03d}"
            f"{suffix}_history.csv"
        )
    )

    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

    return history_path


def print_trial_configuration(
    model_name: str,
    trial_number: int,
    configuration_type: str,
    configuration: dict[str, object],
    number_of_parameters: int,
    trial_seed: int,
) -> None:
    print("\n" + "=" * 72)
    print(
        f"Trial {trial_number:03d} "
        f"({configuration_type})"
    )

    if model_name == "mlp":
        print(
            f"Hidden sizes:         "
            f"{configuration['hidden_sizes']}"
        )

    elif model_name == "gru":
        print(
            f"Hidden size:          "
            f"{configuration['hidden_size']}"
        )
        print(
            f"Number of layers:     "
            f"{configuration['number_of_layers']}"
        )
        print(
            f"Dense size:           "
            f"{configuration['dense_size']}"
        )
        print(
            f"Gradient clip norm:   "
            f"{configuration['gradient_clip_norm']}"
        )

    elif model_name == "transformer":
        print(
            f"d_model:              "
            f"{configuration['d_model']}"
        )
        print(
            f"Attention heads:      "
            f"{configuration['number_of_heads']}"
        )
        print(
            f"Encoder layers:       "
            f"{configuration['number_of_layers']}"
        )
        print(
            f"Feedforward size:     "
            f"{configuration['dim_feedforward']}"
        )
        print(
            f"Gradient clip norm:   "
            f"{configuration['gradient_clip_norm']}"
        )

    print(
        f"Dropout:              "
        f"{configuration['dropout']}"
    )
    print(
        f"Batch size:           "
        f"{configuration['batch_size']}"
    )
    print(
        f"Learning rate:        "
        f"{configuration['learning_rate']}"
    )
    print(
        f"Trainable parameters: "
        f"{number_of_parameters:,}"
    )
    print(f"Trial seed:           {trial_seed}")
    print("=" * 72)


def build_trial_result(
    model_name: str,
    trial_number: int,
    configuration_type: str,
    configuration: dict[str, object],
    number_of_parameters: int,
    best_epoch: int,
    best_train_loss: float,
    best_validation_loss: float,
    epochs_completed: int,
    trial_duration: float,
    trial_seed: int,
    checkpoint_path: Path,
    history_path: Path,
) -> dict[str, object]:
    result: dict[str, object] = {
        "trial_number": trial_number,
        "configuration_type": configuration_type,
        "dropout": float(configuration["dropout"]),
        "batch_size": int(configuration["batch_size"]),
        "learning_rate": float(
            configuration["learning_rate"]
        ),
        "number_of_parameters": number_of_parameters,
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "best_validation_loss": best_validation_loss,
        "best_validation_rmse": float(
            np.sqrt(best_validation_loss)
        ),
        "epochs_completed": epochs_completed,
        "training_time_seconds": trial_duration,
        "random_seed": trial_seed,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
    }

    if model_name == "mlp":
        result["hidden_sizes"] = str(
            tuple(configuration["hidden_sizes"])
        )

    elif model_name == "gru":
        result["hidden_size"] = int(
            configuration["hidden_size"]
        )
        result["number_of_layers"] = int(
            configuration["number_of_layers"]
        )
        result["dense_size"] = int(
            configuration["dense_size"]
        )
        result["gradient_clip_norm"] = float(
            configuration["gradient_clip_norm"]
        )

    elif model_name == "transformer":
        result["d_model"] = int(
            configuration["d_model"]
        )
        result["number_of_heads"] = int(
            configuration["number_of_heads"]
        )
        result["number_of_layers"] = int(
            configuration["number_of_layers"]
        )
        result["dim_feedforward"] = int(
            configuration["dim_feedforward"]
        )
        result["gradient_clip_norm"] = float(
            configuration["gradient_clip_norm"]
        )

    return result


def run_trial(
    model_name: str,
    trial_number: int,
    configuration_type: str,
    configuration: dict[str, object],
    number_of_features: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    trial_seed = RANDOM_SEED + trial_number

    set_random_seed(trial_seed)

    batch_size = int(
        configuration["batch_size"]
    )

    learning_rate = float(
        configuration["learning_rate"]
    )

    weight_decay = float(
        configuration.get(
            "weight_decay",
            0.0,
        )
    )

    gradient_clip_norm = get_gradient_clip_norm(
        model_name=model_name,
        configuration=configuration,
    )

    train_loader = create_dataloader(
        X=X_train,
        y=y_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    validation_loader = create_dataloader(
        X=X_validation,
        y=y_validation,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = create_model(
        model_name=model_name,
        number_of_features=number_of_features,
        configuration=configuration,
    ).to(device)

    criterion = nn.MSELoss()

    if model_name == "transformer":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS,
        eta_min=1e-6,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print_trial_configuration(
        model_name=model_name,
        trial_number=trial_number,
        configuration_type=configuration_type,
        configuration=configuration,
        number_of_parameters=number_of_parameters,
        trial_seed=trial_seed,
    )

    best_validation_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0

    epochs_without_improvement = 0

    checkpoint_path: Path | None = None

    history: list[
        dict[str, float | int]
    ] = []

    trial_start = time.perf_counter()

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        epoch_start = time.perf_counter()

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=gradient_clip_norm,
        )

        validation_loss = evaluate_one_epoch(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step()

        epoch_duration = (
            time.perf_counter()
            - epoch_start
        )

        improvement = (
            best_validation_loss
            - validation_loss
        )

        improved = (
            improvement
            > MINIMUM_IMPROVEMENT
        )

        if improved:
            best_validation_loss = validation_loss
            best_train_loss = train_loss
            best_epoch = epoch

            epochs_without_improvement = 0

            checkpoint_path = save_trial_checkpoint(
                model=model,
                optimizer=optimizer,
                model_name=model_name,
                trial_number=trial_number,
                configuration_type=configuration_type,
                configuration=configuration,
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                number_of_features=number_of_features,
                trial_seed=trial_seed,
            )
        else:
            epochs_without_improvement += 1

        current_learning_rate = float(
        optimizer.param_groups[0]["lr"]
        )   

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_rmse": float(
                    np.sqrt(
                        validation_loss
                    )
                ),
                "epoch_duration_seconds": (
                    epoch_duration
                ),
                "is_best": int(
                    improved
                ),
            }
        )

        marker = (
            " <- best"
            if improved
            else ""
        )

        print(
            f"Epoch {epoch:>3}/{MAX_EPOCHS} | "
            f"train={train_loss:.6f} | "
            f"validation={validation_loss:.6f} | "
            f"validation_rmse="
            f"{np.sqrt(validation_loss):.6f} | "
            f"no_improvement="
            f"{epochs_without_improvement:>2}/"
            f"{EARLY_STOPPING_PATIENCE} | "
            f"time={epoch_duration:.2f}s"
            f"{marker}"
        )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "Early stopping triggered."
            )
            break

    trial_duration = (
        time.perf_counter()
        - trial_start
    )

    history_path = save_trial_history(
        model_name=model_name,
        trial_number=trial_number,
        configuration_type=configuration_type,
        history=history,
    )

    if checkpoint_path is None:
        raise RuntimeError(
            "No checkpoint was saved for "
            f"trial {trial_number}."
        )

    return build_trial_result(
        model_name=model_name,
        trial_number=trial_number,
        configuration_type=configuration_type,
        configuration=configuration,
        number_of_parameters=number_of_parameters,
        best_epoch=best_epoch,
        best_train_loss=best_train_loss,
        best_validation_loss=best_validation_loss,
        epochs_completed=len(history),
        trial_duration=trial_duration,
        trial_seed=trial_seed,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )


def save_search_results(
    model_name: str,
    results: list[dict[str, object]],
) -> Path:
    SEARCH_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        SEARCH_DIRECTORY / f"{model_name}_trials.csv"
    )

    results_frame = pd.DataFrame(results)

    results_frame = results_frame.sort_values(
        by="best_validation_loss",
        ascending=True,
    ).reset_index(drop=True)

    results_frame.insert(
        0,
        "rank",
        np.arange(1, len(results_frame) + 1),
    )

    results_frame.to_csv(
        results_path,
        index=False,
    )

    return results_path


def copy_best_checkpoint(
    model_name: str,
    best_result: dict[str, object],
) -> Path:
    CHECKPOINT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_path = Path(
        str(best_result["checkpoint_path"])
    )

    destination_path = (
        CHECKPOINT_DIRECTORY / f"best_{model_name}.pt"
    )

    shutil.copy2(source_path, destination_path)

    return destination_path


def print_best_configuration(
    model_name: str,
    best_result: dict[str, object],
) -> None:
    if model_name == "mlp":
        print(
            f"Hidden sizes:          "
            f"{best_result['hidden_sizes']}"
        )

    elif model_name == "gru":
        print(
            f"Hidden size:           "
            f"{best_result['hidden_size']}"
        )
        print(
            f"Number of layers:      "
            f"{best_result['number_of_layers']}"
        )
        print(
            f"Dense size:            "
            f"{best_result['dense_size']}"
        )
        print(
            f"Gradient clip norm:    "
            f"{best_result['gradient_clip_norm']}"
        )

    elif model_name == "transformer":
        print(
            f"d_model:               "
            f"{best_result['d_model']}"
        )
        print(
            f"Attention heads:       "
            f"{best_result['number_of_heads']}"
        )
        print(
            f"Encoder layers:        "
            f"{best_result['number_of_layers']}"
        )
        print(
            f"Feedforward size:      "
            f"{best_result['dim_feedforward']}"
        )
        print(
            f"Gradient clip norm:    "
            f"{best_result['gradient_clip_norm']}"
        )

    print(
        f"Dropout:               "
        f"{best_result['dropout']}"
    )
    print(
        f"Batch size:            "
        f"{best_result['batch_size']}"
    )
    print(
        f"Learning rate:         "
        f"{best_result['learning_rate']}"
    )


def main() -> None:
    arguments = parse_arguments()

    model_name = arguments.model.lower()
    number_of_random_trials = arguments.trials

    set_random_seed(RANDOM_SEED)

    device = select_device()

    print("Experiment configuration:")
    print(f"Model:                {model_name}")
    print(f"Device:               {device}")
    print("Baseline trials:      1")
    print(
        f"Random trials:        "
        f"{number_of_random_trials}"
    )
    print(
        f"Total trials:         "
        f"{number_of_random_trials + 1}"
    )
    print(f"Maximum epochs:       {MAX_EPOCHS}")
    print(
        f"Early stopping:       "
        f"{EARLY_STOPPING_PATIENCE}"
    )
    print(
        f"Minimum improvement:  "
        f"{MINIMUM_IMPROVEMENT}"
    )
    print(f"Sequence length:      {SEQUENCE_LENGTH}")
    print(f"Forecast horizon:     {FORECAST_HORIZON}")
    print(f"Experiment seed:      {RANDOM_SEED}")

    if device.type == "cuda":
        print(
            f"GPU:                  "
            f"{torch.cuda.get_device_name(0)}"
        )

    (
        train_data,
        validation_data,
        test_data,
        feature_columns,
    ) = load_processed_data()

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = create_data_splits(
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        feature_columns=feature_columns,
    )

    print("\nDataset shapes:")
    print(f"X_train:       {X_train.shape}")
    print(f"y_train:       {y_train.shape}")
    print(f"X_validation:  {X_validation.shape}")
    print(f"y_validation:  {y_validation.shape}")
    print(f"X_test:        {X_test.shape}")
    print(f"y_test:        {y_test.shape}")

    random_configurations = sample_random_configurations(
        model_name=model_name,
        number_of_trials=number_of_random_trials,
        random_seed=RANDOM_SEED,
    )

    trial_definitions: list[
        tuple[int, str, dict[str, object]]
    ] = [
        (
            0,
            "baseline",
            get_baseline_configuration(model_name),
        )
    ]

    for index, configuration in enumerate(
        random_configurations,
        start=1,
    ):
        trial_definitions.append(
            (
                index,
                "random",
                configuration,
            )
        )

    results: list[dict[str, object]] = []

    experiment_start = time.perf_counter()

    for (
        trial_number,
        configuration_type,
        configuration,
    ) in trial_definitions:
        result = run_trial(
            model_name=model_name,
            trial_number=trial_number,
            configuration_type=configuration_type,
            configuration=configuration,
            number_of_features=len(feature_columns),
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            device=device,
        )

        results.append(result)

    experiment_duration = (
        time.perf_counter() - experiment_start
    )

    sorted_results = sorted(
        results,
        key=lambda item: float(
            item["best_validation_loss"]
        ),
    )

    best_result = sorted_results[0]

    results_path = save_search_results(
        model_name=model_name,
        results=results,
    )

    best_checkpoint_path = copy_best_checkpoint(
        model_name=model_name,
        best_result=best_result,
    )

    baseline_result = next(
        result
        for result in results
        if result["configuration_type"]
        == "baseline"
    )

    baseline_validation_loss = float(
        baseline_result["best_validation_loss"]
    )

    best_validation_loss = float(
        best_result["best_validation_loss"]
    )

    validation_improvement = (
        baseline_validation_loss
        - best_validation_loss
    )

    relative_improvement = (
        validation_improvement
        / baseline_validation_loss
        * 100.0
    )

    print("\n" + "=" * 72)
    print("Experiment finished")
    print("=" * 72)
    print(
        f"Total duration:        "
        f"{experiment_duration:.2f}s"
    )
    print(
        f"Baseline validation:   "
        f"{baseline_validation_loss:.6f}"
    )
    print(
        f"Best validation:       "
        f"{best_validation_loss:.6f}"
    )
    print(
        f"Best validation RMSE:  "
        f"{float(best_result['best_validation_rmse']):.6f}"
    )
    print(
        f"Absolute improvement:  "
        f"{validation_improvement:.6f}"
    )
    print(
        f"Relative improvement:  "
        f"{relative_improvement:.2f}%"
    )
    print(
        f"Best trial:            "
        f"{best_result['trial_number']}"
    )
    print(
        f"Best configuration:    "
        f"{best_result['configuration_type']}"
    )

    print_best_configuration(
        model_name=model_name,
        best_result=best_result,
    )

    print(
        f"Best epoch:            "
        f"{best_result['best_epoch']}"
    )
    print(f"Results:               {results_path}")
    print(
        f"Best checkpoint:       "
        f"{best_checkpoint_path}"
    )


if __name__ == "__main__":
    main()