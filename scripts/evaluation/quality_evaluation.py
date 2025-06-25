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
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Dict, List, Union

from tqdm import tqdm

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.participant.participant import DialogueParticipant
from dialoguekit.utils.dialogue_reader import json_to_dialogues
from scripts.evaluation.rubrics.quality_rubrics import QualityRubrics
from usersimcrs.simulator.llm.interfaces.ollama_interface import (
    OllamaLLMInterface,
)

_PROMPT_EVAL_INTRO = (
    "You are an evaluator and you need to judge how does the "
    "ASSISTANT perform based on the following CONVERSATION HISTORY. Please "
    "rate the ASSISTANT's performance based on the following GRADING RUBRIC.\n"
    "\nCONVERSATION HISTORY:"
)
_PROMPT_EVAL_OUTPUT_FORMAT = (
    'Your output need be a be in a JSON format as follows:\n{"score": '
    '<score>, "score_explanation": <explanation>}\nDo not include '
    "additional information.\n"
)


@dataclass
class QualityScore:
    conversation_id: str
    score: int
    explanation: str = ""

    def to_dict(self) -> Dict[str, Union[int, str]]:
        """Converts the score to a dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "score": self.score,
            "score_explanation": self.explanation,
        }


class QualityScoreEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, QualityScore):
            return o.to_dict()
        return super().default(o)


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
        help="Path to the OLLAMA config file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="(optional) Path to the output file.",
    )
    return parser.parse_args()


def get_prompt(grading_rubric: QualityRubrics, dialogue: Dialogue) -> str:
    """Prepares prompt given grading rubric and dialogue.

    Args:
        grading_rubric: Grading rubric for the aspect.
        dialogue: Dialogue.

    Returns:
        Prompt comprising task definition, grading rubric, and dialogue.
    """
    prompt = _PROMPT_EVAL_INTRO

    # Add dialogue history
    for utterance in dialogue.utterances:
        role = (
            "USER"
            if utterance.participant == DialogueParticipant.USER
            else "ASSISTANT"
        )
        prompt += f"\n{role}: {utterance.text}"

    prompt += f"\n\nGRADING RUBRIC:\n{grading_rubric.value}\n"
    prompt += _PROMPT_EVAL_OUTPUT_FORMAT
    return prompt


if __name__ == "__main__":
    args = parse_args()

    # Load dialogues
    dialogues = json_to_dialogues(args.dialogues)

    # Ollama interface
    ollama_interface = OllamaLLMInterface(
        args.ollama_config, default_response=""
    )

    # Evaluate dialogues
    scores: Dict[str, Dict[str, List[QualityScore]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for dialogue in tqdm(dialogues):
        for aspect in QualityRubrics:
            prompt = get_prompt(aspect, dialogue)
            response = ollama_interface.get_llm_response(prompt)
            try:
                response = response.replace("\\", "\\\\")
                response = json.loads(response)
                score = QualityScore(
                    conversation_id=dialogue.conversation_id,
                    score=int(response["score"]),
                    explanation=response["score_explanation"],
                )
                scores[dialogue.agent_id][aspect.name].append(score)
            except Exception as e:
                print(
                    f"Failed to get score for {aspect} dialogue "
                    f"{dialogue.conversation_id}: {e}\nResponse: {response}"
                )

    # Save scores
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(scores, f, indent=2, cls=QualityScoreEncoder)

    # Summary
    for agent_id, agent_scores in scores.items():
        print(f"Scores for agent {agent_id}:")
        for aspect, scores in agent_scores.items():
            print(f"Aspect: {aspect}")
            avg_score = mean([score.score for score in scores])
            std_dev = stdev([score.score for score in scores])
            print(f"Average score: {avg_score:.2f} (std dev: {std_dev:.2f})")
