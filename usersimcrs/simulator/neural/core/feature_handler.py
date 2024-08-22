"""Interface to build feature vector for neural-based user simulator."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch
from dialoguekit.core.annotated_utterance import AnnotatedUtterance

FeatureVector = Union[torch.Tensor, List[int]]
FeatureMask = Union[torch.Tensor, List[bool]]


class FeatureHandler(ABC):
    @abstractmethod
    def build_input_vector(
        self, utterance: AnnotatedUtterance, **kwargs
    ) -> Tuple[FeatureVector, FeatureMask]:
        """Builds the input vector for a given utterance.

        Args:
            utterance: Annotated utterance.
            kwargs: Additional arguments.

        Raises:
            NotImplementedError: If not implemented in derived class.

        Returns:
            Input vector and mask.
        """
        raise NotImplementedError
