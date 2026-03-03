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
https://arxiv.org/abs/2510.05624
"""

import argparse
import json
from collections import defaultdict
from typing import Dict

from dialoguekit.utils.dialogue_reader import json_to_dialogues
from scripts.evaluation.utility_metric import UtilityMetric


def get_summary(
    scores: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    """Displays a summary of the utility evaluation.

    Args:
        scores: Agent_id -> conversation_id -> utility metrics dict.
    """
    summary: dict = defaultdict(
        lambda: {"total_dialogues": 0, "success_rate": 0, "srrr": 0, "rdl": 0}
    )
    for agent_id, agent_scores in scores.items():
        for conv_metrics in agent_scores.values():
            summary[agent_id]["total_dialogues"] += 1
            summary[agent_id]["success_rate"] += conv_metrics["success"]
            summary[agent_id]["srrr"] += conv_metrics[
                "successful_recommendation_round_ratio"
            ]
            summary[agent_id]["rdl"] += conv_metrics[
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

    metric = UtilityMetric(args.user_nlu_config, args.agent_nlu_config)
    scores = metric.evaluate_agents(
        dialogues,
        recommendation_intent_labels=args.recommendation_intent_labels,
        acceptance_intent_labels=args.accept_intent_labels,
        rejection_intent_labels=args.reject_intent_labels,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(scores, f, indent=2)

    get_summary(scores)
