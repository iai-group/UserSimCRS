"""Reward-per-Dialogue-Length metric implementation.

Evaluates the ratio of accepted recommendations to total dialogue length.
"""

from typing import Any, List, Optional

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.participant.participant import DialogueParticipant
from usersimcrs.evaluation.dialogue_annotation import (
    resolve_intents,
)

from usersimcrs.evaluation.utility_base import DEFAULT_ACC_LABELS, UtilityBase


class RewardPerDialogueLengthMetric(UtilityBase):
    def __init__(
        self,
        name: str = "reward_per_dialogue_length",
    ) -> None:
        """Initializes the reward-per-dialogue-length metric.

        Args:
            name: Metric name.
        """
        super().__init__(name)

    def evaluate_dialogue(
        self,
        dialogue: Dialogue,
        acceptance_intent_labels: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> float:
        """Computes the reward-per-dialogue-length score.

        Args:
            dialogue: Dialogue to evaluate.
            acceptance_intent_labels: Labels for acceptance intents.
                Defaults to ``["ACC"]``.

        Returns:
            Ratio of accepted recommendations to total utterances.
        """
        self._annotate_if_needed(dialogue)
        acc = resolve_intents(acceptance_intent_labels, DEFAULT_ACC_LABELS)
        nb_accepted = sum(
            1
            for utterance in dialogue.utterances
            if utterance.participant == DialogueParticipant.USER
            and any(intent in acc for intent in utterance.get_intents())
        )
        return nb_accepted / len(dialogue.utterances)
