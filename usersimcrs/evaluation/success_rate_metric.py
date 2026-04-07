"""Success Rate metric implementation.

Evaluates whether at least one recommendation was accepted during a dialogue.
"""

from typing import Any, List

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.intent import Intent

from usersimcrs.evaluation.base_metric import BaseMetric
from usersimcrs.evaluation.dialogue_annotation import (
    ensure_dialogue_is_annotated,
    get_recommendation_rounds,
    is_recommendation_accepted,
)


class SuccessRateMetric(BaseMetric):
    def __init__(
        self,
        name: str = "success_rate",
    ) -> None:
        """Initializes the success rate metric.

        Args:
            name: Metric name.
        """
        super().__init__(name)

    def evaluate_dialogue(
        self,
        dialogue: Dialogue,
        recommendation_intents: List[Intent],
        acceptance_intents: List[Intent],
        rejection_intents: List[Intent],
        **kwargs: Any,
    ) -> float:
        """Computes the success rate for a single dialogue.

        Args:
            dialogue: Dialogue to evaluate.
            recommendation_intents: Intents that indicate recommendation.
            acceptance_intents: Intents that indicate acceptance.
            rejection_intents: Intents that indicate rejection.

        Returns:
            1.0 if at least one recommendation was accepted, 0.0 otherwise.
        """
        ensure_dialogue_is_annotated(dialogue)
        rounds = get_recommendation_rounds(dialogue, recommendation_intents)
        return float(
            any(
                is_recommendation_accepted(
                    r, acceptance_intents, rejection_intents
                )
                for r in rounds
            )
        )
