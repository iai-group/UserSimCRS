"""Utility metric class implementation.

Computes three  utility metrics:

- Success Rate (SR)
- Successful Recommendation Round Ratio (SRRR)
- Reward-per-Dialogue-Length (RDL)
"""

from typing import Any, List, Optional, Tuple

from confuse import Configuration

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.intent import Intent
from dialoguekit.nlu.nlu import NLU
from dialoguekit.participant.participant import DialogueParticipant

from usersimcrs.evaluation.base_metric import BaseMetric
from usersimcrs.utils.simulation_utils import get_NLU


class UtilityMetricBase(BaseMetric):
    def __init__(
        self,
        user_nlu_config_path: str,
        agent_nlu_config_path: str,
        name: str,
    ):
        super().__init__(name)
        self.user_nlu_config_path = user_nlu_config_path
        self.agent_nlu_config_path = agent_nlu_config_path
        self._user_nlu: Optional[NLU] = None
        self._agent_nlu: Optional[NLU] = None

    def _annotate_dialogue(
        self, dialogue: Dialogue, user_nlu: NLU, agent_nlu: NLU
    ) -> Dialogue:
        """Annotates utterances with dialogue acts.

        Args:
            dialogue: Dialogue to be annotated.
            user_nlu: User NLU module.
            agent_nlu: Agent NLU module.

        Returns:
            Annotated dialogue.
        """
        for i, utterance in enumerate(dialogue.utterances):
            if not isinstance(utterance, AnnotatedUtterance):
                dialogue.utterances[i] = AnnotatedUtterance.from_utterance(
                    utterance
                )

            if len(utterance.dialogue_acts) > 0:
                continue

            if utterance.participant == DialogueParticipant.USER:
                dialogue.utterances[
                    i
                ].dialogue_acts = user_nlu.extract_dialogue_acts(utterance)
            elif utterance.participant == DialogueParticipant.AGENT:
                dialogue.utterances[
                    i
                ].dialogue_acts = agent_nlu.extract_dialogue_acts(utterance)
            else:
                raise ValueError(
                    f"Unknown participant: {utterance.participant}"
                )
        return dialogue

    def _get_recommendation_rounds(
        self, dialogue: Dialogue, recommendation_intents: List[Intent]
    ) -> List[List[AnnotatedUtterance]]:
        """Gets utterances per recommendation round.

        Args:
            dialogue: Dialogue.
            recommendation_intents: Intents corresponding to recommendation.

        Returns:
            Utterances per recommendation round.
        """
        rounds: List[List[AnnotatedUtterance]] = []
        current_round: List[AnnotatedUtterance] = []
        for utterance in dialogue.utterances:
            if any(
                intent in utterance.get_intents()
                for intent in recommendation_intents
            ):
                if current_round:
                    rounds.append(current_round)
                current_round = [utterance]
            else:
                current_round.append(utterance)
        return rounds

    def _is_recommendation_accepted(
        self,
        round: List[AnnotatedUtterance],
        acceptance_intents: List[Intent],
        rejection_intents: List[Intent],
    ) -> bool:
        """Assesses whether the recommendation was accepted.

        Args:
            round: Utterances in recommendation round.
            acceptance_intents: Intents corresponding to acceptance.
            rejection_intents: Intents corresponding to rejection.

        Returns:
            True if the recommendation was accepted, False otherwise.
        """
        b_accepted = False
        for utterance in round:
            if utterance.participant == DialogueParticipant.USER:
                intents = utterance.get_intents()
                if any(intent in acceptance_intents for intent in intents):
                    b_accepted = True
                elif any(intent in rejection_intents for intent in intents):
                    return False
        return b_accepted

    def _load_nlus(self) -> Tuple[NLU, NLU]:
        """Returns (cached) user and agent NLU modules."""
        if self._user_nlu is None:
            # NLU module for user utterances
            user_nlu_config = Configuration("User NLU Configuration")
            user_nlu_config.set_file(self.user_nlu_config_path)
            self._user_nlu = get_NLU(user_nlu_config)
        if self._agent_nlu is None:
            # NLU module for agent utterances
            agent_nlu_config = Configuration("Agent NLU Configuration")
            agent_nlu_config.set_file(self.agent_nlu_config_path)
            self._agent_nlu = get_NLU(agent_nlu_config)
        return self._user_nlu, self._agent_nlu

    def _get_intent_lists(self, **kwargs: Any) -> Tuple[List[Intent], ...]:
        """Builds intent lists from kwargs."""
        rec_labels = kwargs.get(
            "recommendation_intent_labels", ["REC-S", "REC-E"]
        )
        acc_labels = kwargs.get("acceptance_intent_labels", ["ACC"])
        rej_labels = kwargs.get("rejection_intent_labels", ["REJ"])
        return (
            [Intent(label) for label in rec_labels],
            [Intent(label) for label in acc_labels],
            [Intent(label) for label in rej_labels],
        )

    def _prepare(
        self, dialogue: Dialogue, **kwargs: Any
    ) -> Tuple[Dialogue, List[Intent], List[Intent], List[Intent]]:
        """Annotates dialogue.

        Returns:
            dialogue
            rec_intents: Recommendation intents.
            acc_intents: Acceptance intents.
            rej_intents: Rejection intents.
        """
        user_nlu, agent_nlu = self._load_nlus()
        self._annotate_dialogue(dialogue, user_nlu, agent_nlu)
        rec, acc, rej = self._get_intent_lists(**kwargs)
        return dialogue, rec, acc, rej

    def evaluate_dialogue(self, dialogue: Dialogue, **kwargs: Any) -> float:
        """Computes the metric for a single dialogue."""
        raise NotImplementedError()


class SuccessRateMetric(UtilityMetricBase):
    def __init__(
        self,
        user_nlu_config_path: str,
        agent_nlu_config_path: str,
        name: str = "success_rate",
    ):
        super().__init__(user_nlu_config_path, agent_nlu_config_path, name)

    def _assess_dialogue(
        self,
        dialogue: Dialogue,
        recommendation_intents: List[Intent],
        acceptance_intents: List[Intent],
        rejection_intents: List[Intent],
    ) -> int:
        """Returns number of successful recommendation rounds."""
        rounds = self._get_recommendation_rounds(
            dialogue, recommendation_intents
        )
        return sum(
            1
            for round_utterances in rounds
            if self._is_recommendation_accepted(
                round_utterances, acceptance_intents, rejection_intents
            )
        )

    def evaluate_dialogue(self, dialogue: Dialogue, **kwargs: Any) -> float:
        dlg, rec, acc, rej = self._prepare(dialogue, **kwargs)
        successful_rounds = self._assess_dialogue(dlg, rec, acc, rej)
        return float(successful_rounds > 0)


class SuccessfulRecommendationRoundRatioMetric(UtilityMetricBase):
    def __init__(
        self,
        user_nlu_config_path: str,
        agent_nlu_config_path: str,
        name: str = "successful_recommendation_round_ratio",
    ):
        super().__init__(user_nlu_config_path, agent_nlu_config_path, name)

    def _assess_dialogue(
        self,
        dialogue: Dialogue,
        recommendation_intents: List[Intent],
        acceptance_intents: List[Intent],
        rejection_intents: List[Intent],
    ) -> Tuple[int, int]:
        """Returns successful rounds and total rounds."""
        rounds = self._get_recommendation_rounds(
            dialogue, recommendation_intents
        )
        successful_rounds = sum(
            1
            for round_utterances in rounds
            if self._is_recommendation_accepted(
                round_utterances, acceptance_intents, rejection_intents
            )
        )
        return successful_rounds, len(rounds)

    def evaluate_dialogue(self, dialogue: Dialogue, **kwargs: Any) -> float:
        dlg, rec, acc, rej = self._prepare(dialogue, **kwargs)
        successful_rounds, total_rounds = self._assess_dialogue(
            dlg, rec, acc, rej
        )
        return successful_rounds / total_rounds if total_rounds > 0 else 0.0


class RewardPerDialogueLengthMetric(UtilityMetricBase):
    def __init__(
        self,
        user_nlu_config_path: str,
        agent_nlu_config_path: str,
        name: str = "reward_per_dialogue_length",
    ):
        super().__init__(user_nlu_config_path, agent_nlu_config_path, name)

    def _assess_dialogue(
        self, dialogue: Dialogue, acceptance_intents: List[Intent]
    ) -> Tuple[int, int]:
        """Returns accepted recommendations and dialogue length."""
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
        dlg, _, acc, _ = self._prepare(dialogue, **kwargs)
        nb_accepted_recommendations, dialogue_length = self._assess_dialogue(
            dlg, acc
        )
        return nb_accepted_recommendations / dialogue_length
