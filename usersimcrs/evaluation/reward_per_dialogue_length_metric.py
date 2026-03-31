"""Reward-per-Dialogue-Length metric implementation.

Evaluates the ratio of accepted recommendations to total dialogue length.
"""

from typing import Any, List

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.intent import Intent
from dialoguekit.participant.participant import DialogueParticipant

from usersimcrs.evaluation.base_metric import BaseMetric
from usersimcrs.evaluation.dialogue_annotation import (
    ensure_dialogue_is_annotated,
)


class RewardPerDialogueLengthMetric(BaseMetric):
    def __init__(
        self,
        name: str = "reward_per_dialogue_length",
    ) -> None:
        """Initializes the reward-per-dialogue-length metric.

        Args:
            name: Metric name. Defaults to "reward_per_dialogue_length".
        """
        super().__init__(name)

    def evaluate_dialogue(
        self,
        dialogue: Dialogue,
        acceptance_intents: List[Intent],
        **kwargs: Any,
    ) -> float:
        """Computes the reward-per-dialogue-length score.

        Args:
            dialogue: Dialogue to evaluate.
            acceptance_intents: Acceptance intents.

        Returns:
            Ratio of accepted recommendations to total utterances.
        """
        ensure_dialogue_is_annotated(dialogue)
        nb_accepted = sum(
            1
            for utterance in dialogue.utterances
            if utterance.participant == DialogueParticipant.USER
            and any(
                intent in acceptance_intents
                for intent in utterance.get_intents()
            )
        )
        return nb_accepted / len(dialogue.utterances)
