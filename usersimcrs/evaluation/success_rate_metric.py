"""Success Rate metric implementation.

Evaluates whether at least one recommendation was accepted during a dialogue.
"""

from typing import Any, List, Optional

from dialoguekit.core.dialogue import Dialogue

from usersimcrs.evaluation.base_metric import BaseMetric
from usersimcrs.evaluation.dialogue_annotation import (
    DEFAULT_ACC_LABELS,
    DEFAULT_REC_LABELS,
    DEFAULT_REJ_LABELS,
    annotate_if_needed,
    get_recommendation_rounds,
    is_recommendation_accepted,
    resolve_intents,
)


class SuccessRateMetric(BaseMetric):
    def __init__(
        self,
        user_nlu_config_path: Optional[str] = None,
        agent_nlu_config_path: Optional[str] = None,
        name: str = "success_rate",
    ) -> None:
        """Initializes the success rate metric.

        Args:
            user_nlu_config_path: Path to user NLU configuration.
            agent_nlu_config_path: Path to agent NLU configuration.
            name: Metric name.
        """
        super().__init__(name)
        self._user_nlu_config_path = user_nlu_config_path
        self._agent_nlu_config_path = agent_nlu_config_path

    def evaluate_dialogue(
        self,
        dialogue: Dialogue,
        recommendation_intent_labels: Optional[List[str]] = None,
        acceptance_intent_labels: Optional[List[str]] = None,
        rejection_intent_labels: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> float:
        """Computes the success rate for a single dialogue.

        Args:
            dialogue: Dialogue to evaluate.
            recommendation_intent_labels: Labels for recommendation intents.
                Defaults to ``["REC-S", "REC-E"]``.
            acceptance_intent_labels: Labels for acceptance intents.
                Defaults to ``["ACC"]``.
            rejection_intent_labels: Labels for rejection intents.
                Defaults to ``["REJ"]``.

        Returns:
            1.0 if at least one recommendation was accepted, 0.0 otherwise.
        """
        annotate_if_needed(
            dialogue,
            self._user_nlu_config_path,
            self._agent_nlu_config_path,
        )
        rec = resolve_intents(recommendation_intent_labels, DEFAULT_REC_LABELS)
        acc = resolve_intents(acceptance_intent_labels, DEFAULT_ACC_LABELS)
        rej = resolve_intents(rejection_intent_labels, DEFAULT_REJ_LABELS)
        rounds = get_recommendation_rounds(dialogue, rec)
        return float(
            any(is_recommendation_accepted(r, acc, rej) for r in rounds)
        )
