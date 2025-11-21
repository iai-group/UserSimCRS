"""Automatic evaluation of dialogues with regards to utility.

The script computes three user-centric utility metrics proposed by Bernard and
Balog (2025):

- Success Rate (SR)
- Successful Recommendation Round Ratio (SRRR)
- Reward-per-Dialogue-Length (RDL)

Reference:
Bernard, Nolwenn, and Krisztian Balog. "Limitations of Current Evaluation
Practices for Conversational Recommender Systems and the Potential of User
Simulation." arXiv preprint arXiv:2510.05624 (2025).
"""

import argparse
from collections import defaultdict
import json
from typing import Dict, List, Tuple

from confuse import Configuration
from tqdm import tqdm

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.intent import Intent
from dialoguekit.nlu.nlu import NLU
from dialoguekit.participant.participant import DialogueParticipant
from dialoguekit.utils.dialogue_reader import json_to_dialogues
from usersimcrs.utils.simulation_utils import get_NLU


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
    return [
        annotate_dialogue(dialogue, user_nlu, agent_nlu)
        for dialogue in tqdm(dialogues)
    ]


def get_recommendation_rounds(
    dialogue: Dialogue, recommendation_intents: List[Intent]
) -> List[List[AnnotatedUtterance]]:
    """Gets utterances per recommendation round.

    Args:
        dialogue: Dialogue.
        recommendation_intents: Intents corresponding to recommendation.

    Returns:
        Utterances per recommendation round.
    """
    rounds = []
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


def is_recommendation_accepted(
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
    rounds = get_recommendation_rounds(dialogue, recommendation_intents)
    successful_rounds = 0
    for round in rounds:
        if is_recommendation_accepted(
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


def get_summary(dialogues: List[Dialogue]) -> None:
    """Displays a summary of the utility evaluation.

    Args:
        dialogues: Dialogues.
    """
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


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(prog="utility_evaluation.py")
    parser.add_argument(
        "annotated_dialogues",
        type=str,
        help="Annotated dialogues JSON file.",
    )
    parser.add_argument(
        "user_nlu_config",
        type=str,
        help="User NLU configuration file.",
    )
    parser.add_argument(
        "agent_nlu_config",
        type=str,
        help="Agent NLU configuration file.",
    )
    parser.add_argument(
        "--reject_intent_labels",
        nargs="+",
        default=["REJ"],
        help="Intent labels corresponding to rejection.",
    )
    parser.add_argument(
        "--accept_intent_labels",
        nargs="+",
        default=["ACC"],
        help="Intent labels corresponding to acceptance.",
    )
    parser.add_argument(
        "--recommendation_intent_labels",
        nargs="+",
        default=["REC-S", "REC-E"],
        help="Intent labels corresponding to recommendation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file to save annotated dialogues with utility metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dialogues = json_to_dialogues(args.annotated_dialogues)

    rejection_intents = [Intent(label) for label in args.reject_intent_labels]
    acceptance_intents = [Intent(label) for label in args.accept_intent_labels]
    recommendation_intents = [
        Intent(label) for label in args.recommendation_intent_labels
    ]

    # NLU module for user utterances
    user_nlu_config = Configuration("User NLU Configuration")
    user_nlu_config.set_file(args.user_nlu_config)
    user_nlu = get_NLU(user_nlu_config)

    # NLU module for agent utterances
    agent_nlu_config = Configuration("Agent NLU Configuration")
    agent_nlu_config.set_file(args.agent_nlu_config)
    agent_nlu = get_NLU(agent_nlu_config)

    dialogues = annotate_dialogues(dialogues, user_nlu, agent_nlu)
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
                successful_rounds / total_rounds if total_rounds > 0 else 0.0
            ),
            "reward_per_dialogue_length": (
                nb_accepted_recommendations / len(dialogue.utterances)
            ),
        }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                [dialogue.to_dict() for dialogue in dialogues], f, indent=2
            )

    get_summary(dialogues)
