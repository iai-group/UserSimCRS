"""LLM-based dialogue quality evaluation.

The script evaluates dialogue quality with regards to five aspects:
- Recommendation relevance
- Communication style
- Fluency
- Conversational flow
- Overall satisfaction

Each aspect is scored between 1 and 5, where the scores are described in a
dedicated rubric. The scoring is done using a large language model.
"""

import json
import logging
from typing import Any

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.participant.participant import DialogueParticipant

from usersimcrs.evaluation.base_metric import BaseMetric
from usersimcrs.evaluation.quality_rubrics import QualityRubrics
from usersimcrs.llm_interfaces.llm_interface import LLMInterface


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
        llm_interface: LLMInterface,
        name: str = "quality",
    ) -> None:
        super().__init__(name)
        self.llm_interface = llm_interface

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
            KeyError: When the aspect does not exist in QualityRubrics.
        """
        try:
            aspect_enum = QualityRubrics[aspect]
        except KeyError:
            supported = [e.name for e in QualityRubrics]
            raise KeyError(
                f"Unknown aspect '{aspect}'. Supported aspects: {supported}"
            )
        prompt = self._get_prompt(aspect_enum, dialogue)
        response = self.llm_interface.get_llm_api_response(prompt)
        try:
            response = response.replace("\\", "\\\\")
            response_dict = json.loads(response)
            return float(response_dict["score"])
        except Exception:
            logging.warning(
                "Failed to parse LLM response for %s dialogue %s: %s",
                aspect,
                dialogue.conversation_id,
                response,
            )
            return 0.0
