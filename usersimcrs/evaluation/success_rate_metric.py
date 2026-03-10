"""Success Rate metric implementation.

Evaluates whether at least one recommendation was accepted during a dialogue.
"""

from typing import Any, List, Optional

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.intent import Intent
from dialoguekit.nlu.nlu import NLU

from usersimcrs.evaluation.base_metric import BaseMetric
from usersimcrs.evaluation.dialogue_annotation import (
    get_recommendation_rounds,
    is_recommendation_accepted,
    prepare_dialogue,
)


class SuccessRateMetric(BaseMetric):
    def __init__(
        self,
        user_nlu_config_path: str,
        agent_nlu_config_path: str,
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
        self._user_nlu: Optional[NLU] = None
        self._agent_nlu: Optional[NLU] = None

    def _assess_dialogue(
        self,
        dialogue: Dialogue,
        recommendation_intents: List[Intent],
        acceptance_intents: List[Intent],
        rejection_intents: List[Intent],
    ) -> int:
        """Returns number of successful recommendation rounds.

        Args:
            dialogue: Annotated dialogue.
            recommendation_intents: Intents that signal a recommendation.
            acceptance_intents: Intents that signal acceptance.
            rejection_intents: Intents that signal rejection.

        Returns:
            Number of recommendation rounds that were accepted.
        """
        rounds = get_recommendation_rounds(dialogue, recommendation_intents)
        return sum(
            1
            for round_utterances in rounds
            if is_recommendation_accepted(
                round_utterances, acceptance_intents, rejection_intents
            )
        )

    def evaluate_dialogue(self, dialogue: Dialogue, **kwargs: Any) -> float:
        """Computes the success rate for a single dialogue.

        Args:
            dialogue: Dialogue to evaluate.
            **kwargs: Optional intent label overrides.

        Returns:
            1.0 if at least one recommendation was accepted, 0.0 otherwise.
        """
        dlg, rec, acc, rej, self._user_nlu, self._agent_nlu = prepare_dialogue(
            dialogue,
            self._user_nlu_config_path,
            self._agent_nlu_config_path,
            self._user_nlu,
            self._agent_nlu,
            **kwargs,
        )
        successful_rounds = self._assess_dialogue(dlg, rec, acc, rej)
        return float(successful_rounds > 0)
