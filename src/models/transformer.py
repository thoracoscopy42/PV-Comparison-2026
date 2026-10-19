from __future__ import annotations

import torch
from torch import nn


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        number_of_features: int,
        d_model: int = 128,
        number_of_heads: int = 4,
        number_of_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
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

        if d_model <= 0:
            raise ValueError(
                "d_model must be greater than zero."
            )

        if number_of_heads <= 0:
            raise ValueError(
                "number_of_heads must be greater than zero."
            )

        if d_model % number_of_heads != 0:
            raise ValueError(
                "d_model must be divisible by number_of_heads."
            )

        if number_of_layers <= 0:
            raise ValueError(
                "number_of_layers must be greater than zero."
            )

        if dim_feedforward <= 0:
            raise ValueError(
                "dim_feedforward must be greater than zero."
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
        self.d_model = d_model

        self.input_projection = nn.Linear(
            in_features=number_of_features,
            out_features=d_model,
        )

        self.positional_embedding = nn.Parameter(
            torch.zeros(
                1,
                sequence_length,
                d_model,
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=number_of_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=number_of_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(
                in_features=d_model,
                out_features=d_model,
            ),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=d_model,
                out_features=output_size,
            ),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(
            self.positional_embedding,
            mean=0.0,
            std=0.02,
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

        projected = self.input_projection(X)

        encoded = self.encoder(
            projected + self.positional_embedding
        )

        last_time_step = encoded[:, -1, :]

        return self.regression_head(
            last_time_step
        )