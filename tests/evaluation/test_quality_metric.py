"""Tests for LLM-based quality evaluation."""

from typing import List

import pytest

from dialoguekit.core.dialogue import Dialogue

from usersimcrs.evaluation.quality_metric import QualityMetric
from usersimcrs.evaluation.quality_rubrics import QualityRubrics
from usersimcrs.llm_interfaces.ollama_interface import OllamaLLMInterface


@pytest.fixture
def quality_metric(mock_ollama_interface: OllamaLLMInterface) -> QualityMetric:
    """Provides a QualityMetric instance with a mocked LLM."""
    return QualityMetric(llm_interface=mock_ollama_interface)


def test_evaluate_dialogue_valid_response(
    quality_metric: QualityMetric,
    dialogues: List[Dialogue],
) -> None:
    """Verifies that evaluate_dialogue parses a valid LLM JSON response."""
    quality_metric.llm_interface.get_llm_api_response.return_value = (
        '{"score": 3, "score_explanation": "Average."}'
    )
    for rubric in QualityRubrics:
        score = quality_metric.evaluate_dialogue(
            dialogues[0], aspect=rubric.name
        )
        assert score == 3.0


def test_evaluate_dialogue_unsupported_aspect(
    quality_metric: QualityMetric,
    dialogues: List[Dialogue],
) -> None:
    """Verifies that evaluate_dialogue raises KeyError for an invalid aspect."""
    with pytest.raises(KeyError, match="Unknown aspect 'NOT_A_METRIC'"):
        quality_metric.evaluate_dialogue(dialogues[0], aspect="NOT_A_METRIC")


def test_evaluate_dialogue_missing_score_key(
    quality_metric: QualityMetric,
    dialogues: List[Dialogue],
) -> None:
    """Verifies that evaluate_dialogue returns 0.0 when score is missing."""
    quality_metric.llm_interface.get_llm_api_response.return_value = (
        '{"explanation": "No score field."}'
    )
    score = quality_metric.evaluate_dialogue(dialogues[0], aspect="FLUENCY")
    assert score == 0.0


def test_evaluate_dialogues(
    quality_metric: QualityMetric,
    dialogues: List[Dialogue],
) -> None:
    """Verifies that evaluate_dialogues returns scores by conversation ID."""
    quality_metric.llm_interface.get_llm_api_response.return_value = (
        '{"score": 5, "score_explanation": "Excellent."}'
    )
    results = quality_metric.evaluate_dialogues(dialogues, aspect="OVERALL_SAT")
    assert len(results) == len(dialogues)
    for dialogue in dialogues:
        assert dialogue.conversation_id in results
        assert results[dialogue.conversation_id] == 5.0
