"""Success Rate metric implementation.

Evaluates whether at least one recommendation was accepted during a dialogue.
"""

from typing import Any, List, Optional

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.intent import Intent

from usersimcrs.evaluation.dialogue_annotation import (
    get_recommendation_rounds,
    is_recommendation_accepted,
)
from usersimcrs.evaluation.utility_base_metric import UtilityBaseMetric


class SuccessRateMetric(UtilityBaseMetric):
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
        super().__init__(
            name,
            user_nlu_config_path=user_nlu_config_path,
            agent_nlu_config_path=agent_nlu_config_path,
        )

    def _assess_dialogue(
        self,
        dialogue: Dialogue,
        recommendation_intents: List[Intent],
        acceptance_intents: List[Intent],
        rejection_intents: List[Intent],
    ) -> bool:
        """Checks whether at least one recommendation round was accepted.

        Args:
            dialogue: Annotated dialogue.
            recommendation_intents: Intents that signal a recommendation.
            acceptance_intents: Intents that signal acceptance.
            rejection_intents: Intents that signal rejection.

        Returns:
            True if at least one round was accepted, False otherwise.
        """
        rounds = get_recommendation_rounds(dialogue, recommendation_intents)
        return any(
            is_recommendation_accepted(
                round_utterances, acceptance_intents, rejection_intents
            )
            for round_utterances in rounds
        )

    def evaluate_dialogue(self, dialogue: Dialogue, **kwargs: Any) -> float:
        """Computes the success rate for a single dialogue.

        Args:
            dialogue: Dialogue to evaluate.
            **kwargs: Optional intent label overrides:
                - recommendation_intent_labels: Labels for recommendation
                  intents. Defaults to ["REC-S", "REC-E"].
                - acceptance_intent_labels: Labels for acceptance intents.
                  Defaults to ["ACC"].
                - rejection_intent_labels: Labels for rejection intents.
                  Defaults to ["REJ"].

        Returns:
            1.0 if at least one recommendation was accepted, 0.0 otherwise.
        """
        rec, acc, rej = self._resolve_intents(dialogue=dialogue, **kwargs)
        return float(
            self._assess_dialogue(
                dialogue=dialogue,
                recommendation_intents=rec,
                acceptance_intents=acc,
                rejection_intents=rej,
            )
        )
