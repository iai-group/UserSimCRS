"""Satisfaction metric class implementation.

Wraps DialogueKit's satisfaction classifier into a `BaseMetric` class.
"""

from typing import Any, Optional

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.nlu.models.satisfaction_classifier import (
    SatisfactionClassifierSVM,
)
import argparse

from evaluation.base_metric import BaseMetric


class SatisfactionMetric(BaseMetric):
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
    def parse_args() -> argparse.Namespace:
        """Parses command-line arguments.

        Returns:
            Parsed arguments.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--dialogues",
            type=str,
            required=True,
            help="Path to the dialogues.",
        )
        return parser.parse_args()
