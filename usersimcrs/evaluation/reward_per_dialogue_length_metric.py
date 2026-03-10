"""Reward-per-Dialogue-Length metric implementation.

Evaluates the ratio of accepted recommendations to total dialogue length.
"""

from typing import Any, List, Optional, Tuple

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.intent import Intent
from dialoguekit.participant.participant import DialogueParticipant

from usersimcrs.evaluation.utility_base_metric import UtilityBaseMetric


class RewardPerDialogueLengthMetric(UtilityBaseMetric):
    def __init__(
        self,
        user_nlu_config_path: Optional[str] = None,
        agent_nlu_config_path: Optional[str] = None,
        name: str = "reward_per_dialogue_length",
    ) -> None:
        """Initializes the reward-per-dialogue-length metric.

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
            **kwargs: Optional intent label overrides:
                - recommendation_intent_labels: Labels for recommendation
                  intents. Defaults to ["REC-S", "REC-E"].
                - acceptance_intent_labels: Labels for acceptance intents.
                  Defaults to ["ACC"].
                - rejection_intent_labels: Labels for rejection intents.
                  Defaults to ["REJ"].

        Returns:
            Ratio of accepted recommendations to total utterances.
        """
        _, acc, _ = self._resolve_intents(dialogue=dialogue, **kwargs)
        nb_accepted, dialogue_length = self._assess_dialogue(
            dialogue=dialogue, acceptance_intents=acc
        )
        return nb_accepted / dialogue_length
