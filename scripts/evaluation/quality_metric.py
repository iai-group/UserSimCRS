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
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.participant.participant import DialogueParticipant

from scripts.evaluation.base_metric import BaseMetric
from scripts.evaluation.rubrics.quality_rubrics import QualityRubrics
from usersimcrs.llm_interfaces.ollama_interface import OllamaLLMInterface


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


class QualityMetric(BaseMetric):
    def __init__(
        self,
        ollama_config_path: str,
        default_response: str = "",
        name: str = "quality",
    ) -> None:
        super().__init__(name)
        self.ollama_config_path = ollama_config_path
        self.default_response = default_response
        self._ollama_interface: Optional[OllamaLLMInterface] = None

    @staticmethod
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

    def _get_ollama_interface(self) -> OllamaLLMInterface:
        """Returns Ollama LLM interface."""
        if self._ollama_interface is None:
            self._ollama_interface = OllamaLLMInterface(
                self.ollama_config_path,
                default_response=self.default_response,
            )
        return self._ollama_interface

    def _get_prompt(
        self, grading_rubric: QualityRubrics, dialogue: Dialogue
    ) -> str:
        """Prepares prompt given grading rubric and dialogue.

        Args:
            grading_rubric: Grading rubric for the aspect.
            dialogue: Dialogue.

        Returns:
            Prompt comprising task definition, grading rubric, and dialogue.
        """
        prompt = _PROMPT_EVAL_INTRO
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

    def evaluate_dialogue(
        self, dialogue: Dialogue, aspect: str, **kwargs: Any
    ) -> float:
        """Returns score for a single aspect of a dialogue.

        Args:
            dialogue: Dialogue to evaluate.
            aspect: Aspect to evaluate. Must be one of QualityRubrics enum names

        Returns:
            Score (1-5) for the specified aspect.

        Raises:
            ValueError: When the LLM response cannot be parsed.
        """
        aspect_enum = QualityRubrics[aspect]
        ollama_interface = self._get_ollama_interface()
        prompt = self._get_prompt(aspect_enum, dialogue)
        response = ollama_interface.get_llm_api_response(prompt)
        try:
            response = response.replace("\\", "\\\\")
            response_dict = json.loads(response)
            return float(response_dict["score"])
        except Exception:
            raise ValueError(
                f"Failed to get score for {aspect} dialogue "
                f"{dialogue.conversation_id}: {response}"
            )
