"""Tests for SatisfactionMetric."""

from unittest.mock import MagicMock

import pytest

from dialoguekit.nlu.models.satisfaction_classifier import (
    SatisfactionClassifier,
)
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
    return MagicMock(spec=SatisfactionClassifier)


@pytest.fixture
def satisfaction_metric(mock_classifier):
    """SatisfactionMetric instance with mocked classifier."""
    return SatisfactionMetric(classifier=mock_classifier)


def test_satisfaction_metric_init(mock_classifier) -> None:
    """Test SatisfactionMetric initializes with defaults."""
    metric = SatisfactionMetric(classifier=mock_classifier)
    assert metric.name == "satisfaction"
    assert metric.classifier is mock_classifier


def test_evaluate_dialogue(satisfaction_metric, dialogues) -> None:
    """Test evaluate_dialogue returns classifier score as float."""
    satisfaction_metric.classifier.classify_last_n_dialogue.return_value = 4
    score = satisfaction_metric.evaluate_dialogue(dialogues[0])
    assert score == 4.0
    classify = satisfaction_metric.classifier.classify_last_n_dialogue
    classify.assert_called_once_with(dialogues[0], last_n=None)


def test_evaluate_dialogues(satisfaction_metric, dialogues) -> None:
    """Test evaluate_dialogues returns scores keyed by conversation ID."""
    satisfaction_metric.classifier.classify_last_n_dialogue.return_value = 3
    results = satisfaction_metric.evaluate_dialogues(dialogues)
    assert len(results) == len(dialogues)
    for dialogue in dialogues:
        assert dialogue.conversation_id in results
        assert results[dialogue.conversation_id] == 3.0
