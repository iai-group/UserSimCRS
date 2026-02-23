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

from dialoguekit.utils.dialogue_reader import json_to_dialogues
from scripts.evaluation.utility_metric import UtilityMetric


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
    dialogues = metric.compute(
        dialogues,
        recommendation_intent_labels=args.recommendation_intent_labels,
        acceptance_intent_labels=args.accept_intent_labels,
        rejection_intent_labels=args.reject_intent_labels,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                [dialogue.to_dict() for dialogue in dialogues], f, indent=2
            )

    metric.get_summary(dialogues)
