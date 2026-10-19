from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    gradient_clip_norm: float | None = None,
) -> float:
    model.train()

    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(
            device=device,
            non_blocking=True,
        )

        y_batch = y_batch.to(
            device=device,
            non_blocking=True,
        )

        optimizer.zero_grad()

        predictions = model(X_batch)

        loss = criterion(
            predictions,
            y_batch,
        )

        loss.backward()

        if gradient_clip_norm is not None:
            if gradient_clip_norm <= 0:
                raise ValueError(
                    "gradient_clip_norm must be greater than zero."
                )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )

        optimizer.step()

        batch_size = X_batch.shape[0]

        total_loss += (
            loss.detach().item() * batch_size
        )

        total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "The training DataLoader contains no samples."
        )

    mean_loss = total_loss / total_samples

    return mean_loss


def evaluate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(
                device=device,
                non_blocking=True,
            )

            y_batch = y_batch.to(
                device=device,
                non_blocking=True,
            )

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch,
            )

            batch_size = X_batch.shape[0]

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "The evaluation DataLoader contains no samples."
        )

    mean_loss = total_loss / total_samples

    return mean_loss

def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    all_targets: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(
                device=device,
                non_blocking=True,
            )

            predictions = model(X_batch)

            all_targets.append(
                y_batch.detach().cpu()
            )

            all_predictions.append(
                predictions.detach().cpu()
            )

    if not all_targets:
        raise ValueError(
            "The DataLoader contains no samples."
        )

    targets = torch.cat(
        all_targets,
        dim=0,
    )

    predictions = torch.cat(
        all_predictions,
        dim=0,
    )

    if targets.shape != predictions.shape:
        raise ValueError(
            "Targets and predictions have different shapes. "
            f"Received {tuple(targets.shape)} and "
            f"{tuple(predictions.shape)}."
        )

    if not torch.isfinite(targets).all():
        raise ValueError(
            "Targets contain NaN or infinite values."
        )

    if not torch.isfinite(predictions).all():
        raise ValueError(
            "Predictions contain NaN or infinite values."
        )

    return targets, predictions

