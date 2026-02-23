"""Quality metric class implementation.

Extracted from the original CLI script in `quality_evaluation.py`.
"""

from collections import defaultdict
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tqdm import tqdm

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


@dataclass
class QualityScore:
    conversation_id: str
    score: int
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
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


class QualityMetric(BaseMetric):
    """Quality evaluation metric using an LLM backend.

    The class wraps the prompt construction and LLM calls and returns the
    same structure previously produced by the CLI script:

    { agent_id: { aspect_name: [QualityScore, ...], ... }, ... }
    """

    def __init__(
        self,
        ollama_config_path: str,
        default_response: str = "",
        rubrics: Optional[List[QualityRubrics]] = None,
    ) -> None:
        super().__init__()
        self.ollama_config_path = ollama_config_path
        self.default_response = default_response
        self.rubrics = rubrics or list(QualityRubrics)

    @property
    def name(self) -> str:
        return "quality"

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

    def compute(
        self, dialogues: List[Dialogue], aspects: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, List[QualityScore]]]:
        """Compute quality scores for provided dialogues.

        Args:
            dialogues: list of Dialogue objects
            aspects: optional list of aspect names (strings) to evaluate

        Returns:
            Nested dict: agent_id -> aspect_name -> list[QualityScore]
        """
        ollama_interface = OllamaLLMInterface(
            self.ollama_config_path, default_response=self.default_response
        )

        if aspects:
            aspect_enums = [QualityRubrics[asp] for asp in aspects]
        else:
            aspect_enums = self.rubrics

        scores: Dict[str, Dict[str, List[QualityScore]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for dialogue in tqdm(dialogues):
            for aspect in aspect_enums:
                prompt = self._get_prompt(aspect, dialogue)
                response = ollama_interface.get_llm_api_response(prompt)
                try:
                    response = response.replace("\\", "\\\\")
                    response_dict = json.loads(response)
                    score = QualityScore(
                        conversation_id=dialogue.conversation_id,
                        score=int(response_dict["score"]),
                        explanation=response_dict.get("score_explanation", ""),
                    )
                    scores[dialogue.agent_id][aspect.name].append(score)
                except Exception:
                    print(
                        f"Failed to get score for {aspect} dialogue "
                        f"{dialogue.conversation_id}: {response}"
                    )

        return scores


class RecommendationRelevanceMetric(QualityMetric):
    """Quality metric that evaluates only recommendation relevance."""

    def __init__(self, ollama_config_path: str, default_response: str = ""):
        super().__init__(ollama_config_path, default_response=default_response)
        self.rubrics = [QualityRubrics.REC_RELEVANCE]

    @property
    def name(self) -> str:
        return "quality.recommendation_relevance"


class CommunicationStyleMetric(QualityMetric):
    """Quality metric that evaluates communication style."""

    def __init__(self, ollama_config_path: str, default_response: str = ""):
        super().__init__(ollama_config_path, default_response=default_response)
        self.rubrics = [QualityRubrics.COM_STYLE]

    @property
    def name(self) -> str:
        return "quality.communication_style"


class FluencyMetric(QualityMetric):
    """Quality metric that evaluates fluency."""

    def __init__(self, ollama_config_path: str, default_response: str = ""):
        super().__init__(ollama_config_path, default_response=default_response)
        self.rubrics = [QualityRubrics.FLUENCY]

    @property
    def name(self) -> str:
        return "quality.fluency"


class ConversationalFlowMetric(QualityMetric):
    """Quality metric that evaluates conversational flow."""

    def __init__(self, ollama_config_path: str, default_response: str = ""):
        super().__init__(ollama_config_path, default_response=default_response)
        self.rubrics = [QualityRubrics.CONV_FLOW]

    @property
    def name(self) -> str:
        return "quality.conversational_flow"


class OverallSatisfactionQualityMetric(QualityMetric):
    """Quality metric that evaluates overall satisfaction aspect."""

    def __init__(self, ollama_config_path: str, default_response: str = ""):
        super().__init__(ollama_config_path, default_response=default_response)
        self.rubrics = [QualityRubrics.OVERALL_SAT]

    @property
    def name(self) -> str:
        return "quality.overall_satisfaction"
