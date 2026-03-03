"""Tests for QualityMetric."""

from unittest.mock import MagicMock, patch

import pytest

from dialoguekit.utils.dialogue_reader import json_to_dialogues

from scripts.evaluation.quality_metric import QualityMetric


@pytest.fixture
def dialogues():
    """Load test dialogues."""
    return json_to_dialogues(
        "tests/data/annotated_dialogues.json",
        agent_ids=["Agent"],
        user_ids=["User"],
    )


@pytest.fixture
def mock_ollama():
    """Mock Ollama LLM interface that returns fixed score JSON."""
    interface = MagicMock()
    interface.get_llm_api_response.return_value = (
        '{"score": 4, "score_explanation": "good"}'
    )
    return interface


@pytest.fixture
def metric(mock_ollama):
    """QualityMetric with mocked Ollama interface."""
    with patch.object(
        QualityMetric, "_get_ollama_interface", return_value=mock_ollama
    ):
        yield QualityMetric(ollama_config_path="dummy_config.json")


def test_evaluate_dialogue(
    metric: QualityMetric, mock_ollama, dialogues
) -> None:
    """Test evaluate_dialogue returns mean of aspect scores for a dialogue."""
    dialogue = dialogues[0]
    score = metric.evaluate_dialogue(dialogue)
    assert score == 4.0
    assert mock_ollama.get_llm_api_response.call_count == len(metric.rubrics)


def test_evaluate_dialogue_with_aspects(
    metric: QualityMetric, mock_ollama, dialogues
) -> None:
    """Test evaluate_dialogue with aspects kwarg calls LLM only for aspects."""
    dialogue = dialogues[0]
    aspects = ["REC_RELEVANCE", "FLUENCY"]
    score = metric.evaluate_dialogue(dialogue, aspects=aspects)
    assert score == 4.0
    assert mock_ollama.get_llm_api_response.call_count == 2


def test_evaluate_dialogues(
    metric: QualityMetric, mock_ollama, dialogues
) -> None:
    """Test evaluate_dialogues returns conversation_id -> score."""
    result = metric.evaluate_dialogues(dialogues)
    assert len(result) == len(dialogues)
    for dialogue in dialogues:
        assert dialogue.conversation_id in result
        assert result[dialogue.conversation_id] == 4.0
    expected_calls = len(dialogues) * len(metric.rubrics)
    assert mock_ollama.get_llm_api_response.call_count == expected_calls


def test_evaluate_agents(metric: QualityMetric, dialogues) -> None:
    """Test evaluate_agents returns agent_id -> {conversation_id -> score}."""
    result = metric.evaluate_agents(dialogues)
    assert "Agent" in result
    agent_scores = result["Agent"]
    assert len(agent_scores) == len(dialogues)
    for dialogue in dialogues:
        assert dialogue.conversation_id in agent_scores
        assert agent_scores[dialogue.conversation_id] == 4.0
