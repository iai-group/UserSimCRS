"""Satisfaction metric class implementation.

Satisfaction assessment based on DialogueKit classifier.
"""

from typing import Any

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.nlu.models.satisfaction_classifier import (
    SatisfactionClassifier,
)

from usersimcrs.evaluation.base_metric import BaseMetric


class SatisfactionMetric(BaseMetric):
    def __init__(
        self,
        classifier: SatisfactionClassifier,
        name: str = "satisfaction",
    ) -> None:
        """Initializes the satisfaction metric.

        Args:
            classifier: Satisfaction classifier instance.
            name: Metric name.
        """
        super().__init__(name)
        self.classifier = classifier

    def evaluate_dialogue(self, dialogue: Dialogue, **kwargs: Any) -> float:
        """Computes the satisfaction score for a single dialogue."""
        return float(
            self.classifier.classify_last_n_dialogue(dialogue, last_n=None)
        )
