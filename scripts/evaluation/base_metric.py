from abc import ABC, abstractmethod
from typing import Any

from dialoguekit.core.dialogue import Dialogue


class BaseMetric(ABC):
    """Abstract base class for dialogue evaluation metrics."""

    def __init__(self) -> None:
        """Initialize the metric."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name (e.g., 'quality', 'satisfaction', 'utility')."""
        pass

    @abstractmethod
    def compute(self, dialogues: list[Dialogue], **kwargs: Any) -> Any:
        """Compute the metric over the given dialogues.

        Args:
            dialogues: List of dialogues to compute the metric on.
            **kwargs: Additional arguments specific to the metric.

        Returns:
            Metric scores.
        """
        pass
