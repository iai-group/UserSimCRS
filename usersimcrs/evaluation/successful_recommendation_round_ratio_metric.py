"""Successful Recommendation Round Ratio metric implementation.

Evaluates the ratio of accepted recommendation rounds to total recommendation
rounds in a dialogue.
"""

from typing import Any, List, Optional

from dialoguekit.core.dialogue import Dialogue

from usersimcrs.evaluation.dialogue_annotation import (
    get_recommendation_rounds,
    is_recommendation_accepted,
    resolve_intents,
)
from usersimcrs.evaluation.utility_base import (
    DEFAULT_ACC_LABELS,
    DEFAULT_REC_LABELS,
    DEFAULT_REJ_LABELS,
    UtilityBase,
)


class SuccessfulRecommendationRoundRatioMetric(UtilityBase):
    def __init__(
        self,
        name: str = "successful_recommendation_round_ratio",
    ) -> None:
        """Initializes the successful recommendation round ratio metric.

        Args:
            name: Metric name.
        """
        super().__init__(name)

    def evaluate_dialogue(
        self,
        dialogue: Dialogue,
        recommendation_intent_labels: Optional[List[str]] = None,
        acceptance_intent_labels: Optional[List[str]] = None,
        rejection_intent_labels: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> float:
        """Computes the successful recommendation round ratio.

        Args:
            dialogue: Dialogue to evaluate.
            recommendation_intent_labels: Labels for recommendation intents.
                Defaults to ``["REC-S", "REC-E"]``.
            acceptance_intent_labels: Labels for acceptance intents.
                Defaults to ``["ACC"]``.
            rejection_intent_labels: Labels for rejection intents.
                Defaults to ``["REJ"]``.

        Returns:
            Ratio of accepted recommendation rounds to total rounds,
            or 0.0 if there are no recommendation rounds.
        """
        self._annotate_if_needed(dialogue)
        rec = resolve_intents(recommendation_intent_labels, DEFAULT_REC_LABELS)
        acc = resolve_intents(acceptance_intent_labels, DEFAULT_ACC_LABELS)
        rej = resolve_intents(rejection_intent_labels, DEFAULT_REJ_LABELS)
        rounds = get_recommendation_rounds(dialogue, rec)
        successful = sum(
            1 for r in rounds if is_recommendation_accepted(r, acc, rej)
        )
        return successful / len(rounds) if rounds else 0.0
