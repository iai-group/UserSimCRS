"""Tests for QualityMetric."""

from unittest.mock import MagicMock
import pytest
from dialoguekit.utils.dialogue_reader import json_to_dialogues
from usersimcrs.evaluation.quality_metric import QualityMetric


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
    interface = MagicMock()
    interface.get_llm_api_response.return_value = (
        '{"score": 4, "score_explanation": "good"}'
    )
    return interface


@pytest.fixture
def metric(mock_llm_interface):
    return QualityMetric(llm_interface=mock_llm_interface)


def test_evaluate_dialogue(
    metric: QualityMetric, mock_llm_interface, dialogues
) -> None:
    """Test evaluate_dialogue returns score for REC_RELEVANCE aspect."""
    dialogue = dialogues[0]
    score = metric.evaluate_dialogue(dialogue, aspect="REC_RELEVANCE")
    assert score == 4.0
    assert mock_llm_interface.get_llm_api_response.call_count == 1


def test_evaluate_dialogue_different_aspect(
    metric: QualityMetric, mock_llm_interface, dialogues
) -> None:
    """Test evaluate_dialogue with FLUENCY aspect."""
    dialogue = dialogues[0]
    score = metric.evaluate_dialogue(dialogue, aspect="FLUENCY")
    assert score == 4.0
    assert mock_llm_interface.get_llm_api_response.call_count == 1


def test_evaluate_dialogues(
    metric: QualityMetric, mock_llm_interface, dialogues
) -> None:
    """Test evaluate_dialogues with for COM_STYLE aspect."""
    result = metric.evaluate_dialogues(dialogues, aspect="COM_STYLE")
    assert len(result) == len(dialogues)
    for dialogue in dialogues:
        assert dialogue.conversation_id in result
        assert result[dialogue.conversation_id] == 4.0
    assert mock_llm_interface.get_llm_api_response.call_count == len(dialogues)
