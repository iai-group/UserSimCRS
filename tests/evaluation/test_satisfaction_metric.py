"""Tests for SatisfactionMetric."""

from unittest.mock import MagicMock
import pytest
from dialoguekit.utils.dialogue_reader import json_to_dialogues
from usersimcrs.evaluation.satisfaction_metric import SatisfactionMetric


@pytest.fixture
def dialogues():
    """Load test dialogues."""
    return json_to_dialogues(
        "tests/data/annotated_dialogues.json",
        agent_ids=["Agent"],
        user_ids=["User"],
    )


@pytest.fixture
def mock_classifier():
    """Mock satisfaction classifier."""
    classifier = MagicMock()
    classifier.classify_last_n_dialogue = MagicMock(return_value=3.5)
    return classifier


@pytest.fixture
def metric(mock_classifier):
    return SatisfactionMetric(classifier=mock_classifier)


def test_evaluate_dialogue(metric: SatisfactionMetric, dialogues) -> None:
    """Test evaluate_dialogue for a single dialogue."""
    dialogue = dialogues[0]
    score = metric.evaluate_dialogue(dialogue)
    assert score == 3.5


def test_evaluate_dialogues(
    metric: SatisfactionMetric, mock_classifier, dialogues
) -> None:
    """Test evaluate_dialogues for list of dialogues."""
    result = metric.evaluate_dialogues(dialogues)
    assert len(result) == len(dialogues)
    for dialogue in dialogues:
        assert dialogue.conversation_id in result
        assert result[dialogue.conversation_id] == 3.5
    assert mock_classifier.classify_last_n_dialogue.call_count == len(dialogues)
