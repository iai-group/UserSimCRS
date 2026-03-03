"""Utility metric class implementation.

Encapsulates the logic from `utility_evaluation.py` into a `Metric`.
"""

from typing import Any, Dict, List, Optional, Tuple, cast

from confuse import Configuration

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.intent import Intent
from dialoguekit.nlu.nlu import NLU
from dialoguekit.participant.participant import DialogueParticipant
from usersimcrs.utils.simulation_utils import get_NLU
from scripts.evaluation.base_metric import Metric


class UtilityMetric(Metric):
    """Computes utility metrics for dialogues.

    Constructor takes paths to user and agent NLU configuration files.
    """

    def __init__(
        self,
        user_nlu_config_path: str,
        agent_nlu_config_path: str,
        name: str = "utility",
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

    def _annotate_dialogues(
        self, dialogues: List[Dialogue], user_nlu: NLU, agent_nlu: NLU
    ) -> List[Dialogue]:
        """Annotates dialogues with dialogue acts.

        Args:
            dialogues: Dialogues.
            user_nlu: User NLU module.
            agent_nlu: Agent NLU module.

        Returns:
            Annotated dialogues.
        """
        # TODO: Move this to DialogueKit
        # See: https://github.com/iai-group/UserSimCRS/issues/219
        return [
            self._annotate_dialogue(dialogue, user_nlu, agent_nlu)
            for dialogue in dialogues
        ]

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

    def _assess_dialogue(
        self,
        dialogue: Dialogue,
        recommendation_intents: List[Intent],
        acceptance_intents: List[Intent],
        rejection_intents: List[Intent],
    ) -> Tuple[int, int, int]:
        """Assesses the utility of the dialogue.

        Args:
            dialogue: Dialogue.
            recommendation_intents: Intents corresponding to recommendation.
            acceptance_intents: Intents corresponding to acceptance.
            rejection_intents: Intents corresponding to rejection.

        Returns:
            Tuple of number of accepted recommendations, successful
                recommendation rounds and total recommendation rounds.
        """
        # TODO: Optimize overall assessment to avoid multiple iterations over
        # utterances.
        rounds = self._get_recommendation_rounds(
            dialogue, recommendation_intents
        )
        successful_rounds = 0
        for round in rounds:
            if self._is_recommendation_accepted(
                round, acceptance_intents, rejection_intents
            ):
                successful_rounds += 1

        nb_accepted_recommendations = sum(
            1
            for utterance in dialogue.utterances
            if utterance.participant == DialogueParticipant.USER
            and any(
                intent in acceptance_intents
                for intent in utterance.get_intents()
            )
        )
        return nb_accepted_recommendations, successful_rounds, len(rounds)

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

    def evaluate_dialogues(
        self, dialogues: List[Dialogue], **kwargs: Any
    ) -> Dict[str, Dict[str, float]]:
        """Computes all utility metrics for every dialogue.

        Overrides base to return full metrics dict per dialogue rather than
        a single float, since utility evaluation aggregates SR, SRRR, and RDL.

        Returns:
            conversation_id -> metrics dict with keys: success,
            successful_recommendation_round_ratio, reward_per_dialogue_length.
        """
        return {
            dialogue.conversation_id: self._get_utility_metrics(
                dialogue, **kwargs
            )
            for dialogue in dialogues
        }

    def evaluate_agents(
        self, dialogues: List[Dialogue], **kwargs: Any
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Computes utility metrics per agent, returning full metrics per
        dialogue.

        Returns:
            agent_id -> conversation_id -> metrics dict (success, srrr, rdl).
        """
        result = super().evaluate_agents(dialogues, **kwargs)
        return cast(Dict[str, Dict[str, Dict[str, float]]], result)

    def _get_utility_metrics(
        self, dialogue: Dialogue, **kwargs: Any
    ) -> Dict[str, float]:
        """Returns full utility dict for one dialogue."""
        user_nlu, agent_nlu = self._load_nlus()
        self._annotate_dialogue(dialogue, user_nlu, agent_nlu)
        (
            recommendation_intents,
            acceptance_intents,
            rejection_intents,
        ) = self._get_intent_lists(**kwargs)
        (
            nb_accepted_recommendations,
            successful_rounds,
            total_rounds,
        ) = self._assess_dialogue(
            dialogue,
            recommendation_intents,
            acceptance_intents,
            rejection_intents,
        )
        return {
            "success": float(successful_rounds > 0),
            "successful_recommendation_round_ratio": (
                successful_rounds / total_rounds if total_rounds > 0 else 0.0
            ),
            "reward_per_dialogue_length": (
                nb_accepted_recommendations / len(dialogue.utterances)
                if dialogue.utterances
                else 0.0
            ),
        }

    def evaluate_dialogue(self, dialogue: Dialogue, **kwargs: Any) -> float:
        """Computes one utility metric for a single dialogue.

        Args:
            dialogue: Dialogue to evaluate.
            metric: One of "success", "successful_recommendation_round_ratio",
                "reward_per_dialogue_length". Default "success".

        Returns:
            The selected metric value as float.
        """
        metrics = self._get_utility_metrics(dialogue, **kwargs)
        metric = kwargs.get("metric", "success")
        if metric not in metrics:
            raise ValueError(
                f"Unknown metric '{metric}'. "
                f"Expected one of {list(metrics.keys())}"
            )
        return metrics[metric]
