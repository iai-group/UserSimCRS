"""Encoder-only transformer model for neural user simulator."""

import math

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
        x = x + self.pe[: x.size(0)]  # type: ignore[index]
        return self.dropout(x)


class TransformerEncoderModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        nhead: int,
        hidden_dim: int,
        num_encoder_layers: int,
        num_token: int,
        dropout: float = 0.5,
    ) -> None:
        """Initializes a encoder-only transformer model.

        Args:
            input_dim: Size of the input vector.
            output_dim: Size of the output vector.
            nhead: Number of heads.
            hidden_dim: Hidden dimension.
            num_encoder_layers: Number of encoder layers.
            num_token: Number of tokens in the vocabulary.
            dropout: Dropout rate. Defaults to 0.5.
        """
        super(TransformerEncoderModel, self).__init__()
        self.d_model = input_dim

        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        self.embedding = nn.Embedding(num_token, input_dim)

        # Encoder layers
        norm_layer = nn.LayerNorm(input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=norm_layer,
        )

        self.linear = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

        self.init_weights()

    def init_weights(self) -> None:
        """Initializes weights of the network."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(
        self, src: torch.Tensor, src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Performs forward pass.

        Args:
            src: Source tensor.
            src_mask: Mask tensor.

        Returns:
            Output tensor.
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=src_mask)
        output = self.linear(output)
        return output
