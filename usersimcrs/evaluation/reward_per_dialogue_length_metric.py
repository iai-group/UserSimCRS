"""Reward-per-Dialogue-Length metric implementation.

Evaluates the ratio of accepted recommendations to total dialogue length.
"""

from typing import Any, List, Optional, Tuple

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.intent import Intent
from dialoguekit.nlu.nlu import NLU
from dialoguekit.participant.participant import DialogueParticipant

from usersimcrs.evaluation.base_metric import BaseMetric
from usersimcrs.evaluation.dialogue_annotation import (
    get_intent_lists,
    prepare_dialogue,
)


class RewardPerDialogueLengthMetric(BaseMetric):
    def __init__(
        self,
        user_nlu_config_path: Optional[str] = None,
        agent_nlu_config_path: Optional[str] = None,
        name: str = "reward_per_dialogue_length",
    ) -> None:
        """Initializes the reward-per-dialogue-length metric.

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
        self, dialogue: Dialogue, acceptance_intents: List[Intent]
    ) -> Tuple[int, int]:
        """Returns accepted recommendations and dialogue length.

        Args:
            dialogue: Annotated dialogue.
            acceptance_intents: Intents that signal acceptance.

        Returns:
            Tuple of (accepted_recommendations, dialogue_length).
        """
        nb_accepted_recommendations = sum(
            1
            for utterance in dialogue.utterances
            if utterance.participant == DialogueParticipant.USER
            and any(
                intent in acceptance_intents
                for intent in utterance.get_intents()
            )
        )
        return nb_accepted_recommendations, len(dialogue.utterances)

    def evaluate_dialogue(self, dialogue: Dialogue, **kwargs: Any) -> float:
        """Computes the reward-per-dialogue-length score.

        Args:
            dialogue: Dialogue to evaluate.
            **kwargs: Optional intent label overrides.

        Returns:
            Ratio of accepted recommendations to total utterances.
        """
        if self._user_nlu_config_path and self._agent_nlu_config_path:
            (
                dialogue,
                _,
                acc,
                _,
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
            _, acc, _ = get_intent_lists(**kwargs)

        nb_accepted, dialogue_length = self._assess_dialogue(dialogue, acc)
        return nb_accepted / dialogue_length
