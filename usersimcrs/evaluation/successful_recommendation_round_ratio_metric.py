"""Successful Recommendation Round Ratio metric implementation.

Evaluates the ratio of accepted recommendation rounds to total recommendation
rounds in a dialogue.
"""

from typing import Any, List, Optional, Tuple

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.intent import Intent
from dialoguekit.nlu.nlu import NLU

from usersimcrs.evaluation.base_metric import BaseMetric
from usersimcrs.evaluation.dialogue_annotation import (
    get_intent_lists,
    get_recommendation_rounds,
    is_recommendation_accepted,
    prepare_dialogue,
)


class SuccessfulRecommendationRoundRatioMetric(BaseMetric):
    def __init__(
        self,
        user_nlu_config_path: Optional[str] = None,
        agent_nlu_config_path: Optional[str] = None,
        name: str = "successful_recommendation_round_ratio",
    ) -> None:
        """Initializes the successful recommendation round ratio metric.

        When NLU config paths are provided, dialogues are annotated
        automatically. When omitted, dialogues must be pre-annotated
        (e.g., via :func:`annotate_dialogues`).

        Args:
            user_nlu_config_path: Path to user NLU configuration.
            agent_nlu_config_path: Path to agent NLU configuration.
            name: Metric name.
        """
        super().__init__(name)
        self._user_nlu_config_path = user_nlu_config_path
        self._agent_nlu_config_path = agent_nlu_config_path
        self._user_nlu: Optional[NLU] = None
        self._agent_nlu: Optional[NLU] = None

    def _assess_dialogue(
        self,
        dialogue: Dialogue,
        recommendation_intents: List[Intent],
        acceptance_intents: List[Intent],
        rejection_intents: List[Intent],
    ) -> Tuple[int, int]:
        """Returns successful and total recommendation rounds.

        Args:
            dialogue: Annotated dialogue.
            recommendation_intents: Intents that signal a recommendation.
            acceptance_intents: Intents that signal acceptance.
            rejection_intents: Intents that signal rejection.

        Returns:
            Tuple of (successful_rounds, total_rounds).
        """
        rounds = get_recommendation_rounds(dialogue, recommendation_intents)
        successful_rounds = sum(
            1
            for round_utterances in rounds
            if is_recommendation_accepted(
                round_utterances, acceptance_intents, rejection_intents
            )
        )
        return successful_rounds, len(rounds)

    def evaluate_dialogue(self, dialogue: Dialogue, **kwargs: Any) -> float:
        """Computes the successful recommendation round ratio.

        Args:
            dialogue: Dialogue to evaluate.
            **kwargs: Optional intent label overrides.

        Returns:
            Ratio of accepted recommendation rounds to total rounds,
            or 0.0 if there are no recommendation rounds.
        """
        if self._user_nlu_config_path and self._agent_nlu_config_path:
            (
                dialogue,
                rec,
                acc,
                rej,
                self._user_nlu,
                self._agent_nlu,
            ) = prepare_dialogue(
                dialogue,
                self._user_nlu_config_path,
                self._agent_nlu_config_path,
                self._user_nlu,
                self._agent_nlu,
                **kwargs,
            )
        else:
            rec, acc, rej = get_intent_lists(**kwargs)

        successful_rounds, total_rounds = self._assess_dialogue(
            dialogue, rec, acc, rej
        )
        return successful_rounds / total_rounds if total_rounds > 0 else 0.0
