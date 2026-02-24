"""Abstract base class for dialogue evaluation metrics.

Subclasses implement only compute_score(dialogue, **kwargs). The base class
provides aggregation at three levels: per dialogue, per dialogues, and per
agent.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from dialoguekit.core.dialogue import Dialogue


class BaseMetric(ABC):
    def __init__(self, name: str) -> None:
        """Initializes the metric.

        Args:
            name: Metric name.
        """
        self.name = name

    @abstractmethod
    def compute_score(self, dialogue: Dialogue, **kwargs: Any) -> float:
        """Computes the metric for a single dialogue.

        Subclasses must implement this method.

        Args:
            dialogue: Single dialogue to score.
            **kwargs: Additional arguments specific to the metric.

        Returns:
            Score for the dialogue.

        Raises:
            NotImplementedError: When not implemented by a subclass.
        """
        raise NotImplementedError()

    def compute_scores_for_dialogues(
        self, dialogues: Dict[str, Dialogue], **kwargs: Any
    ) -> Dict[str, float]:
        """Computes the metric for each dialogue in a dict of dialogues.

        Args:
            dialogues: Dict of dialogues
            **kwargs: Passed through to compute_score.

        Returns:
            Dict of scores per dialogue.
        """
        return {
            dialog_id: self.compute_score(dialogue, **kwargs)
            for dialog_id, dialogue in dialogues.items()
        }

    def compute_scores_per_agent(
        self, dialogues_by_agent: Dict[str, Dict[str, Dialogue]], **kwargs: Any
    ) -> Dict[str, Dict[str, float]]:
        """Computes the metric per agent over their dialogues.

        Args:
            dialogues_by_agent: Dict of dialogues per agent.
            **kwargs: Passed through to compute_score.

        Returns:
            Dict of scores per agent.
        """
        return {
            agent_id: self.compute_scores_for_dialogues(dialogues, **kwargs)
            for agent_id, dialogues in dialogues_by_agent.items()
        }
