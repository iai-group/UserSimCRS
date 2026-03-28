"""Tests for LLM-based quality evaluation.

These tests cover:
- valid JSON responses returned by the LLM interface
- evaluation across all supported quality rubric aspects
- fallback behavior when the response payload is missing the score field
- error handling for unsupported aspect names
- batch evaluation over multiple dialogues

The metric depends on an LLM interface, which is mocked here so the tests stay
deterministic and do not require external model calls.
"""

import pytest

from dialoguekit.core.dialogue import Dialogue

from usersimcrs.evaluation.quality_metric import QualityMetric
from usersimcrs.evaluation.quality_rubrics import QualityRubrics


@pytest.fixture
def quality_metric(mock_ollama_interface) -> QualityMetric:
    """QualityMetric instance with mocked LLM."""
    return QualityMetric(llm_interface=mock_ollama_interface)


def test_evaluate_dialogue_valid_response(
    quality_metric: QualityMetric,
    dialogues: list[Dialogue],
) -> None:
    """Test evaluate_dialogue parses a valid LLM JSON response."""
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
    dialogues: list[Dialogue],
) -> None:
    """Test evaluate_dialogue raises KeyError for an unsupported aspect."""
    with pytest.raises(KeyError, match="Unknown aspect 'NOT_A_METRIC'"):
        quality_metric.evaluate_dialogue(dialogues[0], aspect="NOT_A_METRIC")


def test_evaluate_dialogue_missing_score_key(
    quality_metric: QualityMetric,
    dialogues: list[Dialogue],
) -> None:
    """Test evaluate_dialogue returns 0.0 when 'score' key is missing."""
    quality_metric.llm_interface.get_llm_api_response.return_value = (
        '{"explanation": "No score field."}'
    )
    score = quality_metric.evaluate_dialogue(dialogues[0], aspect="FLUENCY")
    assert score == 0.0


def test_evaluate_dialogues(
    quality_metric: QualityMetric,
    dialogues: list[Dialogue],
) -> None:
    """Test evaluate_dialogues returns scores keyed by conversation ID."""
    quality_metric.llm_interface.get_llm_api_response.return_value = (
        '{"score": 5, "score_explanation": "Excellent."}'
    )
    results = quality_metric.evaluate_dialogues(dialogues, aspect="OVERALL_SAT")
    assert len(results) == len(dialogues)
    for dialogue in dialogues:
        assert dialogue.conversation_id in results
        assert results[dialogue.conversation_id] == 5.0
