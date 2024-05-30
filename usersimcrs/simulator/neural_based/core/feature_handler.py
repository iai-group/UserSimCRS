"""Interface to build feature vector for neural-based user simulator."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import joblib
import torch
from dialoguekit.core.annotated_utterance import AnnotatedUtterance


class FeatureHandler(ABC):
    @abstractmethod
    def get_feature_vector(
        self, utterance: AnnotatedUtterance, **kwargs
    ) -> torch.Tensor:
        """Builds a feature vector for a given utterance.

        Args:
            utterance: Annotated utterance.
            kwargs: Additional arguments.

        Raises:
            NotImplementedError: If not implemented in derived class.
        """
        raise NotImplementedError

    @classmethod
    def load_handler(cls, path: str) -> FeatureHandler:
        """Loads feature handler from a file.

        Args:
            path: Path to load feature handler.

        Raises:
            FileNotFoundError: If the file is not found.

        Returns:
            Feature handler.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File '{path}' not found.")

        return joblib.load(path)

    def save_handler(self, path: str) -> None:
        """Saves feature handler to a file.

        Args:
            path: Path to save feature handler.
        """
        joblib.dump(self, path)
