"""Automatic evaluation of dialogues.

This script evaluates dialogues with regards to user satisfaction. It uses
DialogueKit's satisfaction classifier, which assigns a score between 1 and 5.
"""

import argparse
from statistics import mean, stdev
from typing import Dict

from dialoguekit.nlu.models.satisfaction_classifier import (
    SatisfactionClassifierSVM,
)
from dialoguekit.utils.dialogue_reader import json_to_dialogues


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

    # Satisfaction classifier
    satisfaction_classifier = SatisfactionClassifierSVM()

    # Evaluate dialogues
    scores: Dict[str, Dict[str, float]] = defaultdict(dict)

    for i, dialogue in enumerate(dialogues):
        scores[dialogue.agent_id][i] = (
            satisfaction_classifier.classify_last_n_dialogue(
                dialogue, last_n=None
            )
        )

    # Summary
    for agent, agent_scores in scores.items():
        avg_score = mean(agent_scores.values())
        stdev_score = stdev(agent_scores.values())
        max_score = max(agent_scores.values())
        min_score = min(agent_scores.values())
        print(f"Agent: {agent} / Num. dialogues: {len(agent_scores)}")
        print(f"Min score: {min_score}")
        print(f"Max score: {max_score}")
        print(f"Average score: {avg_score:.3f} (stdev: {stdev_score:.3f})")
