"""Quality metric class implementation.

Extracted from the original CLI script in `quality_evaluation.py`.
"""

import json
from statistics import mean
from typing import Any, List, Optional

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.participant.participant import DialogueParticipant

from scripts.evaluation.base_metric import Metric
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


class QualityMetric(Metric):
    """Quality evaluation metric using an LLM backend.

    Returns scores as floats (average across aspects per dialogue).
    """

    def __init__(
        self,
        ollama_config_path: str,
        default_response: str = "",
        rubrics: Optional[List[QualityRubrics]] = None,
        name: str = "quality",
    ) -> None:
        super().__init__(name)
        self.ollama_config_path = ollama_config_path
        self.default_response = default_response
        self.rubrics = rubrics or list(QualityRubrics)
        self._ollama_interface: Optional[OllamaLLMInterface] = None

    def _get_ollama_interface(self) -> OllamaLLMInterface:
        """Returns (cached) Ollama LLM interface."""
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

    def evaluate_dialogue(self, dialogue: Dialogue, **kwargs: Any) -> float:
        """Returns average score across aspects for a single dialogue (1–5)."""
        aspects = kwargs.get("aspects")
        if aspects:
            aspect_enums = [QualityRubrics[asp] for asp in aspects]
        else:
            aspect_enums = self.rubrics

        ollama_interface = self._get_ollama_interface()
        scores: List[float] = []

        for aspect in aspect_enums:
            prompt = self._get_prompt(aspect, dialogue)
            response = ollama_interface.get_llm_api_response(prompt)
            try:
                response = response.replace("\\", "\\\\")
                response_dict = json.loads(response)
                scores.append(int(response_dict["score"]))
            except Exception:
                print(
                    f"Failed to get score for {aspect} dialogue "
                    f"{dialogue.conversation_id}: {response}"
                )

        return mean(scores) if scores else 0.0
