"""Classifiers for intents, slots, and slot-intent mapping."""

import torch
import torch.nn as nn


class MultiIntentClassifier(nn.Module):
    def __init__(
        self, input_dim: int, num_labels: int, dropout_rate: float = 0.0
    ) -> None:
        """Initializes classifier for multiple intents detection.

        Args:
            input_dim: Dimension of input features.
            num_labels: Number of labels.
            dropout_rate: Dropout rate. Defaults to 0.
        """
        super(MultiIntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(input_dim, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass.

        Args:
            x: Input features.

        Returns:
            Sigmoid output for each label.
        """
        x = self.dropout(x)
        x = self.classifier(x)
        return self.sigmoid(x)

    def init_weights(self) -> None:
        """Initializes weights."""
        nn.init.uniform_(self.classifier.weight)
        nn.init.uniform_(self.classifier.bias)


class TagIntentClassifier(nn.Module):
    def __init__(
        self, input_dim: int, num_labels: int, dropout: float = 0.0
    ) -> None:
        """Initializes classifier for slot-intent mapping.

        Args:
            input_dim: Dimension of input features.
            num_labels: Number of labels.
            dropout: Dropout rate. Defaults to 0.
        """
        super(TagIntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass.

        Args:
            x: Input features.

        Returns:
            Softmax output for each label.
        """
        x = self.dropout(x)
        x = self.classifier(x)
        return self.softmax(x)


class TokenClassifier(nn.Module):
    def __init__(
        self, input_dim: int, num_labels: int, dropout_rate: float = 0.0
    ) -> None:
        """Initializes token classifier.

        Args:
            input_dim: Dimension of input features.
            num_labels: Number of labels.
            dropout_rate: Dropout rate. Defaults to 0.
        """
        super(TokenClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass.

        Args:
            x: Input features.

        Returns:
            Logits for each label.
        """
        x = self.dropout(x)
        return self.linear(x)
