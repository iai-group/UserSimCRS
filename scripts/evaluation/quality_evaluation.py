"""Script to evaluate dialogue quality using an LLM.

The script evaluates dialogue quality with regards to five aspects:
- Recommendation relevance
- Communication style
- Fluency
- Conversational flow
- Overall satisfaction

Each aspect is scored between 1 and 5, where the scores are described in a
dedicated rubric. The scoring is done using a large language model.
"""

import argparse
import json
import os
from statistics import mean, stdev
from typing import Dict, List

from dialoguekit.utils.dialogue_reader import json_to_dialogues

from scripts.evaluation.quality_metric import QualityMetric, QualityScoreEncoder


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
    parser.add_argument(
        "--ollama_config",
        type=str,
        required=True,
        help="Path to the Ollama config file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="(optional) Path to the output file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load dialogues
    dialogues = json_to_dialogues(args.dialogues)

    metric = QualityMetric(args.ollama_config)
    scores: Dict[str, Dict[str, List]] = metric.compute(dialogues)

    # Save scores
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(scores, f, indent=2, cls=QualityScoreEncoder)

    # Summary
    for agent_id, agent_scores in scores.items():
        print(f"Scores for agent {agent_id}:")
        for aspect_name, aspect_scores in agent_scores.items():
            print(f"Aspect: {aspect_name}")
            avg_score = mean([score.score for score in aspect_scores])
            std_dev = stdev([score.score for score in aspect_scores])
            print(f"Average score: {avg_score:.2f} (std dev: {std_dev:.2f})")
