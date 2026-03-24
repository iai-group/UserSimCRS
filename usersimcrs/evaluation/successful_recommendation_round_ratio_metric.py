"""Successful Recommendation Round Ratio metric implementation.

Evaluates the ratio of accepted recommendation rounds to total recommendation
rounds in a dialogue.
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


class SuccessfulRecommendationRoundRatioMetric(BaseMetric):
    def __init__(
        self,
        name: str = "successful_recommendation_round_ratio",
    ) -> None:
        """Initializes the successful recommendation round ratio metric.

        Args:
            name: Metric name. Defaults to
              "successful_recommendation_round_ratio".
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
        """Computes the successful recommendation round ratio.

        Args:
            dialogue: Dialogue to evaluate.
            recommendation_intents: Intents that indicate recommendation.
            acceptance_intents: Intents that indicate acceptance.
            rejection_intents: Intents that indicate rejection.

        Returns:
            Ratio of accepted recommendation rounds to total rounds,
              or 0.0 if there are no recommendation rounds.
        """
        ensure_dialogue_is_annotated(dialogue)
        rounds = get_recommendation_rounds(dialogue, recommendation_intents)
        successful = sum(
            1
            for r in rounds
            if is_recommendation_accepted(
                r, acceptance_intents, rejection_intents
            )
        )
        return successful / len(rounds) if rounds else 0.0
