from __future__ import annotations

import torch
from torch import nn


class GRUForecaster(nn.Module):
    def __init__(
        self,
        number_of_features: int,
        hidden_size: int = 128,
        number_of_layers: int = 1,
        dense_size: int = 64,
        dropout: float = 0.0,
        output_size: int = 1,
    ) -> None:
        super().__init__()

        if number_of_features <= 0:
            raise ValueError(
                "number_of_features must be greater than zero."
            )

        if hidden_size <= 0:
            raise ValueError(
                "hidden_size must be greater than zero."
            )

        if number_of_layers <= 0:
            raise ValueError(
                "number_of_layers must be greater than zero."
            )

        if dense_size <= 0:
            raise ValueError(
                "dense_size must be greater than zero."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0, 1)."
            )

        if output_size <= 0:
            raise ValueError(
                "output_size must be greater than zero."
            )

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.number_of_layers = number_of_layers

        gru_dropout = (
            dropout
            if number_of_layers > 1
            else 0.0
        )

        self.gru = nn.GRU(
            input_size=number_of_features,
            hidden_size=hidden_size,
            num_layers=number_of_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        self.regression_head = nn.Sequential(
            nn.Linear(
                in_features=hidden_size,
                out_features=dense_size,
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=dense_size,
                out_features=output_size,
            ),
        )

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        if X.ndim != 3:
            raise ValueError(
                "Expected X with shape "
                "[batch, sequence_length, features], "
                f"received {tuple(X.shape)}."
            )

        if X.shape[2] != self.number_of_features:
            raise ValueError(
                "Incorrect number of features. "
                f"Expected {self.number_of_features}, "
                f"received {X.shape[2]}."
            )

        gru_output, _ = self.gru(X)

        last_time_step = gru_output[:, -1, :]

        predictions = self.regression_head(
            last_time_step
        )

        return predictions