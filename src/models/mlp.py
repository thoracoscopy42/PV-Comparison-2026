from __future__ import annotations

import torch
from torch import nn


class MLPForecaster(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        number_of_features: int,
        hidden_sizes: tuple[int, ...] = (128, 64),
        dropout: float = 0.0,
        output_size: int = 1,
    ) -> None:
        super().__init__()

        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be greater than zero."
            )

        if number_of_features <= 0:
            raise ValueError(
                "number_of_features must be greater than zero."
            )

        if not hidden_sizes:
            raise ValueError(
                "hidden_sizes must contain at least one layer."
            )

        if any(size <= 0 for size in hidden_sizes):
            raise ValueError(
                "Every hidden layer size must be greater than zero."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0, 1)."
            )

        if output_size <= 0:
            raise ValueError(
                "output_size must be greater than zero."
            )

        self.sequence_length = sequence_length
        self.number_of_features = number_of_features
        self.input_size = (
            sequence_length * number_of_features
        )

        layers: list[nn.Module] = []

        current_size = self.input_size

        for hidden_size in hidden_sizes:
            layers.append(
                nn.Linear(
                    in_features=current_size,
                    out_features=hidden_size,
                )
            )

            layers.append(nn.ReLU())

            if dropout > 0.0:
                layers.append(
                    nn.Dropout(p=dropout)
                )

            current_size = hidden_size

        layers.append(
            nn.Linear(
                in_features=current_size,
                out_features=output_size,
            )
        )

        self.network = nn.Sequential(*layers)

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

        if X.shape[1] != self.sequence_length:
            raise ValueError(
                "Incorrect sequence length. "
                f"Expected {self.sequence_length}, "
                f"received {X.shape[1]}."
            )

        if X.shape[2] != self.number_of_features:
            raise ValueError(
                "Incorrect number of features. "
                f"Expected {self.number_of_features}, "
                f"received {X.shape[2]}."
            )

        batch_size = X.shape[0]

        flattened = X.reshape(
            batch_size,
            self.input_size,
        )

        predictions = self.network(flattened)

        return predictions