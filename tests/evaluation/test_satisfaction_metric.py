"""Tests for classifier-based satisfaction evaluation.

These tests cover:
- conversion of classifier outputs to float satisfaction scores
- support for both integer and fractional classifier predictions
- batch evaluation over multiple dialogues

The metric depends on a satisfaction classifier, which is mocked here so the
tests remain deterministic and do not require external model inference.
"""

from unittest.mock import MagicMock

import pytest

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.nlu.models.satisfaction_classifier import (
    SatisfactionClassifier,
)

from usersimcrs.evaluation.satisfaction_metric import SatisfactionMetric


@pytest.fixture
def mock_classifier() -> MagicMock:
    """Mock satisfaction classifier."""
    return MagicMock(spec=SatisfactionClassifier)


@pytest.fixture
def satisfaction_metric(mock_classifier: MagicMock) -> SatisfactionMetric:
    """SatisfactionMetric instance with mocked classifier."""
    return SatisfactionMetric(classifier=mock_classifier)


@pytest.mark.parametrize(
    ("classifier_output", "expected_score"),
    [(4, 4.0), (3.7, pytest.approx(3.7))],
)
def test_evaluate_dialogue(
    satisfaction_metric: SatisfactionMetric,
    dialogues: list[Dialogue],
    classifier_output: float,
    expected_score: float,
) -> None:
    """Test evaluate_dialogue returns classifier output as float."""
    satisfaction_metric.classifier.classify_last_n_dialogue.return_value = (
        classifier_output
    )
    score = satisfaction_metric.evaluate_dialogue(dialogues[0])
    assert score == expected_score
    classify = satisfaction_metric.classifier.classify_last_n_dialogue
    classify.assert_called_once_with(dialogues[0], last_n=None)


def test_evaluate_dialogues(
    satisfaction_metric: SatisfactionMetric,
    dialogues: list[Dialogue],
) -> None:
    """Test evaluate_dialogues returns scores keyed by conversation ID."""
    satisfaction_metric.classifier.classify_last_n_dialogue.return_value = 3
    results = satisfaction_metric.evaluate_dialogues(dialogues)
    assert len(results) == len(dialogues)
    for dialogue in dialogues:
        assert dialogue.conversation_id in results
        assert results[dialogue.conversation_id] == 3.0
