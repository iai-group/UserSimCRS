"""Satisfaction metric class implementation.

Wraps DialogueKit's satisfaction classifier into a `Metric` class.
"""

from statistics import mean, stdev
from typing import Any, Dict, Optional

from dialoguekit.core.dialogue import Dialogue  # type: ignore
from dialoguekit.nlu.models.satisfaction_classifier import (
    SatisfactionClassifierSVM,
)

from scripts.evaluation.base_metric import Metric


class SatisfactionMetric(Metric):
    """Wraps the `SatisfactionClassifierSVM` to compute satisfaction scores."""

    def __init__(
        self,
        classifier: Optional[SatisfactionClassifierSVM] = None,
        name: str = "satisfaction",
    ):
        super().__init__(name)
        self.classifier = classifier or SatisfactionClassifierSVM()

    def evaluate_dialogue(self, dialogue: Dialogue, **kwargs: Any) -> float:
        """Computes the satisfaction score for a single dialogue."""
        return self.classifier.classify_last_n_dialogue(dialogue, last_n=None)

    @staticmethod
    def get_average(agent_scores: Dict[str, float]) -> float:
        """Returns the average score for an agent's dialogues."""
        return mean(agent_scores.values()) if agent_scores else 0.0

    @staticmethod
    def get_stdev(agent_scores: Dict[str, float]) -> float:
        """Returns the standard deviation of scores for an agent's dialogues."""
        if len(agent_scores) < 2:
            return 0.0
        return stdev(agent_scores.values())

    @staticmethod
    def get_max(agent_scores: Dict[str, float]) -> float:
        """Returns the maximum score for an agent's dialogues."""
        return max(agent_scores.values()) if agent_scores else 0.0

    @staticmethod
    def get_min(agent_scores: Dict[str, float]) -> float:
        """Returns the minimum score for an agent's dialogues."""
        return min(agent_scores.values()) if agent_scores else 0.0
