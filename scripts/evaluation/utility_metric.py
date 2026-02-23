"""Utility metric class implementation.

Encapsulates the logic from `utility_evaluation.py` into a `BaseMetric`.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

from confuse import Configuration

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.intent import Intent
from dialoguekit.nlu.nlu import NLU
from dialoguekit.participant.participant import DialogueParticipant
from usersimcrs.utils.simulation_utils import get_NLU
from scripts.evaluation.base_metric import BaseMetric


def annotate_dialogue(
    dialogue: Dialogue, user_nlu: NLU, agent_nlu: NLU
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
            raise ValueError(f"Unknown participant: {utterance.participant}")
    return dialogue


def annotate_dialogues(
    dialogues: List[Dialogue], user_nlu: NLU, agent_nlu: NLU
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
        annotate_dialogue(dialogue, user_nlu, agent_nlu)
        for dialogue in dialogues
    ]


def _get_recommendation_rounds(
    dialogue: Dialogue, recommendation_intents: List[Intent]
) -> List[List[AnnotatedUtterance]]:
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
    round: List[AnnotatedUtterance],
    acceptance_intents: List[Intent],
    rejection_intents: List[Intent],
) -> bool:
    b_accepted = False
    for utterance in round:
        if utterance.participant == DialogueParticipant.USER:
            intents = utterance.get_intents()
            if any(intent in acceptance_intents for intent in intents):
                b_accepted = True
            elif any(intent in rejection_intents for intent in intents):
                return False
    return b_accepted


def assess_dialogue(
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
        Tuple of number of accepted recommendations, successful recommendation
          rounds and total recommendation rounds.
    """
    # TODO: Optimize overall assessment to avoid multiple iterations over
    # utterances.
    rounds = _get_recommendation_rounds(dialogue, recommendation_intents)
    successful_rounds = 0
    for round in rounds:
        if _is_recommendation_accepted(
            round, acceptance_intents, rejection_intents
        ):
            successful_rounds += 1

    nb_accepted_recommendations = sum(
        1
        for utterance in dialogue.utterances
        if utterance.participant == DialogueParticipant.USER
        and any(
            intent in acceptance_intents for intent in utterance.get_intents()
        )
    )
    return nb_accepted_recommendations, successful_rounds, len(rounds)


class UtilityMetric(BaseMetric):
    """Computes utility metrics for dialogues.

    Constructor takes paths to user and agent NLU configuration files.
    """

    def __init__(self, user_nlu_config_path: str, agent_nlu_config_path: str):
        super().__init__()
        self.user_nlu_config_path = user_nlu_config_path
        self.agent_nlu_config_path = agent_nlu_config_path

    @property
    def name(self) -> str:
        return "utility"

    def _load_nlus(self) -> Tuple[NLU, NLU]:
        user_nlu_config = Configuration("User NLU Configuration")
        user_nlu_config.set_file(self.user_nlu_config_path)
        user_nlu = get_NLU(user_nlu_config)

        agent_nlu_config = Configuration("Agent NLU Configuration")
        agent_nlu_config.set_file(self.agent_nlu_config_path)
        agent_nlu = get_NLU(agent_nlu_config)

        return user_nlu, agent_nlu

    def compute(
        self,
        dialogues: List[Dialogue],
        recommendation_intent_labels: List[str] = ["REC-S", "REC-E"],
        acceptance_intent_labels: List[str] = ["ACC"],
        rejection_intent_labels: List[str] = ["REJ"],
    ) -> List[Dialogue]:
        user_nlu, agent_nlu = self._load_nlus()

        dialogues = annotate_dialogues(dialogues, user_nlu, agent_nlu)

        recommendation_intents = [
            Intent(label) for label in recommendation_intent_labels
        ]
        acceptance_intents = [
            Intent(label) for label in acceptance_intent_labels
        ]
        rejection_intents = [Intent(label) for label in rejection_intent_labels]

        for dialogue in dialogues:
            (
                nb_accepted_recommendations,
                successful_rounds,
                total_rounds,
            ) = assess_dialogue(
                dialogue,
                recommendation_intents,
                acceptance_intents,
                rejection_intents,
            )
            dialogue.metadata["utility"] = {
                "success": int(successful_rounds > 0),
                "successful_recommendation_round_ratio": (
                    successful_rounds / total_rounds
                    if total_rounds > 0
                    else 0.0
                ),
                "reward_per_dialogue_length": (
                    nb_accepted_recommendations / len(dialogue.utterances)
                    if len(dialogue.utterances) > 0
                    else 0.0
                ),
            }

        return dialogues

    def get_summary(self, dialogues: List[Dialogue]) -> None:
        summary: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "total_dialogues": 0,
                "success_rate": 0,
                "srrr": 0,
                "rdl": 0,
            }
        )
        for dialogue in dialogues:
            summary[dialogue.agent_id]["total_dialogues"] += 1
            summary[dialogue.agent_id]["success_rate"] += dialogue.metadata[
                "utility"
            ]["success"]
            summary[dialogue.agent_id]["srrr"] += dialogue.metadata["utility"][
                "successful_recommendation_round_ratio"
            ]
            summary[dialogue.agent_id]["rdl"] += dialogue.metadata["utility"][
                "reward_per_dialogue_length"
            ]

        for agent_id, stats in summary.items():
            total = stats["total_dialogues"]
            print(f"Agent: {agent_id}")
            print(f"\tTotal Dialogues: {total}")
            print(f"\tSuccess Rate: {stats['success_rate'] / total:.4f}")
            print(
                "\tSuccessful Recommendation Round Ratio: "
                f"{stats['srrr'] / total:.4f}"
            )
            print(f"\tReward-per-Dialogue-Length: {stats['rdl'] / total:.4f}")
            print()


class UtilitySuccessMetric(UtilityMetric):
    """Extracts per-dialogue success flag from utility analysis."""

    @property
    def name(self) -> str:
        return "utility.success"

    def compute(self, dialogues: List[Dialogue], *args, **kwargs):
        dialogues = super().compute(dialogues, *args, **kwargs)

        results: Dict[str, Dict[int, int]] = defaultdict(dict)
        for i, dialogue in enumerate(dialogues):
            results[dialogue.agent_id][i] = int(
                dialogue.metadata.get("utility", {}).get("success", 0)
            )
        return results


class UtilitySRRRMetric(UtilityMetric):
    """Extracts successful recommendation round ratio per dialogue."""

    @property
    def name(self) -> str:
        return "utility.successful_recommendation_round_ratio"

    def compute(self, dialogues: List[Dialogue], *args, **kwargs):
        dialogues = super().compute(dialogues, *args, **kwargs)

        results: Dict[str, Dict[int, float]] = defaultdict(dict)
        for i, dialogue in enumerate(dialogues):
            results[dialogue.agent_id][i] = float(
                dialogue.metadata.get("utility", {}).get(
                    "successful_recommendation_round_ratio", 0.0
                )
            )
        return results


class UtilityRDLMetric(UtilityMetric):
    """Extracts reward-per-dialogue-length per dialogue."""

    @property
    def name(self) -> str:
        return "utility.reward_per_dialogue_length"

    def compute(self, dialogues: List[Dialogue], *args, **kwargs):
        dialogues = super().compute(dialogues, *args, **kwargs)

        results: Dict[str, Dict[int, float]] = defaultdict(dict)
        for i, dialogue in enumerate(dialogues):
            results[dialogue.agent_id][i] = float(
                dialogue.metadata.get("utility", {}).get(
                    "reward_per_dialogue_length", 0.0
                )
            )
        return results
