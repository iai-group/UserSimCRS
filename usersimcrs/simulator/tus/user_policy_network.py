"""User policy network for the Transformer-based User Simulator (TUS).

From reference: "The model structure includes a linear layer and position
encoding for inputs, two transformer layers, and one linear layer for outputs."
"""

import math
from typing import Any, Dict

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        **kwargs,
    ) -> None:
        """Initializes positional encoding layer.

        Args:
            d_model: Dimension of the model.
            dropout: Dropout rate. Defaults to 0.1.
            max_len: Maximum length of the input sequence. Defaults to 5000.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass.

        Args:
            x: Input tensor.

        Returns:
            Positional encoded tensor.
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class UserPolicyNetwork(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes the user policy network.

        Args:
            config: Configuration of the neural network.
        """
        super(UserPolicyNetwork, self).__init__()

        self._config = config

        # Linear layer and position encoding for inputs
        self._embedding_dim = config.get("embedding_dim")
        self_hidden_dim = config.get("hidden_dim")

        self.embedding_layer = nn.Linear(self._embedding_dim, self_hidden_dim)
        self.positional_encoding = PositionalEncoding(
            self_hidden_dim, **self._config
        )

        # Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._embedding_dim,
            nhead=self._config.get("nhead"),
            dim_feedforward=self._hidden_dim,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self._config.get("num_encoder_layers"),
        )

        # Linear layer for outputs
        self._output_dim = config.get("output_dim")
        self.fc = nn.Linear(self_hidden_dim, self._output_dim)

        self.init_weights()

    def init_weights(self) -> None:
        """Initializes weights of the network."""
        initrange = 0.1
        self.embedding_layer.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(
        self,
        input_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Performs forward pass.

        Args:
            input_features: Input features.
            mask: Mask tensor.

        Returns:
            Output tensor.
        """
        src = self.embedding_layer(input_features) * math.sqrt(
            self._hidden_dim
        )
        src = self.positional_encoding(src)
        if mask is not None:
            mask = nn.Transformer.generate_square_subsequent_mask(len(src))
        src = self.encoder(src, mask)
        src = self.fc(src)
        return src
