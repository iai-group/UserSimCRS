"""Interface to build feature vector for neural-based user simulator."""

from abc import ABC, abstractmethod

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
