"""Automatic evaluation of dialogues.

This script evaluates dialogues with regards to user satisfaction. It uses
DialogueKit's satisfaction classifier, which assigns a score between 1 and 5.
"""

import argparse

from dialoguekit.utils.dialogue_reader import json_to_dialogues
from scripts.evaluation.satisfaction_metric import SatisfactionMetric


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dialogues",
        type=str,
        required=True,
        help="Path to the dialogues.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load dialogues
    dialogues = json_to_dialogues(args.dialogues)
    print(f"Loaded {len(dialogues)} dialogues.")

    metric = SatisfactionMetric()
    scores = metric.evaluate_agents(dialogues)

    # Summary
    for agent, agent_scores in scores.items():
        avg_score = metric.get_average(agent_scores)
        stdev_score = metric.get_stdev(agent_scores)
        max_score = metric.get_max(agent_scores)
        min_score = metric.get_min(agent_scores)
        print(f"Agent: {agent} / Num. dialogues: {len(agent_scores)}")
        print(f"Min score: {min_score}")
        print(f"Max score: {max_score}")
        print(f"Average score: {avg_score:.3f} (stdev: {stdev_score:.3f})")
