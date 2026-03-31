"""Tests for classifier-based satisfaction evaluation."""

from typing import List
from unittest.mock import MagicMock

import pytest

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.nlu.models.satisfaction_classifier import (
    SatisfactionClassifier,
)

from usersimcrs.evaluation.satisfaction_metric import SatisfactionMetric


@pytest.fixture
def mock_classifier() -> MagicMock:
    """Provides a mocked satisfaction classifier."""
    return MagicMock(spec=SatisfactionClassifier)


@pytest.fixture
def satisfaction_metric(mock_classifier: MagicMock) -> SatisfactionMetric:
    """Provides a SatisfactionMetric instance with a mocked classifier."""
    return SatisfactionMetric(classifier=mock_classifier)


@pytest.mark.parametrize(
    ("classifier_output", "expected_score"),
    [(4, 4.0), (3.7, pytest.approx(3.7))],
)
def test_evaluate_dialogue(
    satisfaction_metric: SatisfactionMetric,
    dialogues: List[Dialogue],
    classifier_output: float,
    expected_score: float,
) -> None:
    """Verifies that evaluate_dialogue returns the classifier output."""
    satisfaction_metric.classifier.classify_last_n_dialogue.return_value = (
        classifier_output
    )
    score = satisfaction_metric.evaluate_dialogue(dialogues[0])
    assert score == expected_score
    classify = satisfaction_metric.classifier.classify_last_n_dialogue
    classify.assert_called_once_with(dialogues[0], last_n=None)


def test_evaluate_dialogues(
    satisfaction_metric: SatisfactionMetric,
    dialogues: List[Dialogue],
) -> None:
    """Verifies that evaluate_dialogues returns scores by conversation ID."""
    satisfaction_metric.classifier.classify_last_n_dialogue.return_value = 3
    results = satisfaction_metric.evaluate_dialogues(dialogues)
    assert len(results) == len(dialogues)
    for dialogue in dialogues:
        assert dialogue.conversation_id in results
        assert results[dialogue.conversation_id] == 3.0
