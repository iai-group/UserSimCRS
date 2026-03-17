"""Tests for QualityMetric."""

from unittest.mock import MagicMock

import pytest

from usersimcrs.evaluation.quality_metric import QualityMetric
from usersimcrs.llm_interfaces.llm_interface import LLMInterface


@pytest.fixture
def mock_llm_interface():
    """Mock LLM interface."""
    return MagicMock(spec=LLMInterface)


@pytest.fixture
def quality_metric(mock_llm_interface):
    """QualityMetric instance with mocked LLM."""
    return QualityMetric(llm_interface=mock_llm_interface)


def test_evaluate_dialogue_valid_response(quality_metric) -> None:
    """Test evaluate_dialogue parses a valid LLM JSON response."""
    quality_metric.llm_interface.get_llm_api_response.return_value = (
        '{"score": 4, "score_explanation": "Good."}'
    )
    dialogue = MagicMock()
    dialogue.utterances = []
    dialogue.conversation_id = "cid"
    score = quality_metric.evaluate_dialogue(dialogue, aspect="FLUENCY")
    assert score == 4.0


def test_evaluate_dialogue_invalid_json_returns_zero(quality_metric) -> None:
    """Test evaluate_dialogue returns 0.0 on unparseable model output."""
    quality_metric.llm_interface.get_llm_api_response.return_value = "not json"
    dialogue = MagicMock()
    dialogue.utterances = []
    dialogue.conversation_id = "cid"
    score = quality_metric.evaluate_dialogue(dialogue, aspect="FLUENCY")
    assert score == 0.0
