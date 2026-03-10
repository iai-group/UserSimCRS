"""Tests for QualityMetric."""

from unittest.mock import MagicMock

import pytest

from dialoguekit.utils.dialogue_reader import json_to_dialogues

from usersimcrs.evaluation.quality_metric import QualityMetric
from usersimcrs.evaluation.quality_rubrics import QualityRubrics
from usersimcrs.llm_interfaces.llm_interface import LLMInterface


@pytest.fixture
def dialogues():
    """Load test dialogues."""
    return json_to_dialogues(
        "tests/data/annotated_dialogues.json",
        agent_ids=["Agent"],
        user_ids=["User"],
    )


@pytest.fixture
def mock_llm_interface():
    """Mock LLM interface."""
    return MagicMock(spec=LLMInterface)


@pytest.fixture
def quality_metric(mock_llm_interface):
    """QualityMetric instance with mocked LLM."""
    return QualityMetric(llm_interface=mock_llm_interface)


def test_quality_metric_init(mock_llm_interface) -> None:
    """Test QualityMetric initializes with correct name and LLM."""
    metric = QualityMetric(llm_interface=mock_llm_interface)
    assert metric.name == "quality"
    assert metric.llm_interface is mock_llm_interface


def test_quality_metric_custom_name(mock_llm_interface) -> None:
    """Test QualityMetric accepts custom name."""
    metric = QualityMetric(llm_interface=mock_llm_interface, name="custom")
    assert metric.name == "custom"


def test_get_prompt(quality_metric, dialogues) -> None:
    """Test _get_prompt builds prompt with dialogue and rubric."""
    dialogue = dialogues[0]
    prompt = quality_metric._get_prompt(QualityRubrics.FLUENCY, dialogue)

    assert "CONVERSATION HISTORY" in prompt
    assert "USER: Utterance 1" in prompt
    assert "ASSISTANT: Utterance 2" in prompt
    assert "GRADING RUBRIC" in prompt
    assert "Fluency" in prompt
    assert '{"score"' in prompt


def test_get_prompt_all_aspects(quality_metric, dialogues) -> None:
    """Test _get_prompt works for every quality aspect."""
    dialogue = dialogues[0]
    for aspect in QualityRubrics:
        prompt = quality_metric._get_prompt(aspect, dialogue)
        assert aspect.value in prompt


def test_evaluate_dialogue_valid_response(quality_metric, dialogues) -> None:
    """Test evaluate_dialogue parses a valid LLM JSON response."""
    quality_metric.llm_interface.get_llm_api_response.return_value = (
        '{"score": 4, "score_explanation": "Good fluency."}'
    )
    score = quality_metric.evaluate_dialogue(dialogues[0], aspect="FLUENCY")
    assert score == 4.0


def test_evaluate_dialogue_all_aspects(quality_metric, dialogues) -> None:
    """Test evaluate_dialogue succeeds for each aspect name."""
    quality_metric.llm_interface.get_llm_api_response.return_value = (
        '{"score": 3, "score_explanation": "Average."}'
    )
    for aspect in QualityRubrics:
        score = quality_metric.evaluate_dialogue(
            dialogues[0], aspect=aspect.name
        )
        assert score == 3.0


def test_evaluate_dialogue_missing_score_key(quality_metric, dialogues) -> None:
    """Test evaluate_dialogue returns 0.0 when 'score' key is missing."""
    quality_metric.llm_interface.get_llm_api_response.return_value = (
        '{"explanation": "No score field."}'
    )
    score = quality_metric.evaluate_dialogue(dialogues[0], aspect="FLUENCY")
    assert score == 0.0


def test_evaluate_dialogue_unknown_aspect(quality_metric, dialogues) -> None:
    """Test evaluate_dialogue raises KeyError for unsupported aspect."""
    with pytest.raises(KeyError, match="Unknown aspect"):
        quality_metric.evaluate_dialogue(dialogues[0], aspect="NONEXISTENT")


def test_evaluate_dialogues(quality_metric, dialogues) -> None:
    """Test evaluate_dialogues returns scores keyed by conversation ID."""
    quality_metric.llm_interface.get_llm_api_response.return_value = (
        '{"score": 5, "score_explanation": "Excellent."}'
    )
    results = quality_metric.evaluate_dialogues(dialogues, aspect="OVERALL_SAT")
    assert len(results) == len(dialogues)
    for dialogue in dialogues:
        assert dialogue.conversation_id in results
        assert results[dialogue.conversation_id] == 5.0
