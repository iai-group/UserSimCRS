"""Abstract base class for dialogue evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Any, List

from dialoguekit.core.dialogue import Dialogue


class BaseMetric(ABC):
    def __init__(self, name: str):
        """Initializes the metric.

        Args:
            name: Metric name (e.g., 'quality', 'satisfaction', 'utility').
        """
        self.name = name

    @abstractmethod
    def compute(self, dialogues: List[Dialogue], **kwargs: Any) -> Any:
        """Computes the metric over the given dialogues.

        Args:
            dialogues: List of dialogues to compute the metric on.
            **kwargs: Additional arguments specific to the metric.

        Returns:
            Metric result; shape is defined by the concrete metric.
        """
        raise NotImplementedError()
