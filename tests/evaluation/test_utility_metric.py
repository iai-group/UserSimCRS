"""Tests for UtilityMetric."""

from unittest.mock import patch

import pytest

from dialoguekit.utils.dialogue_reader import json_to_dialogues

from scripts.evaluation.utility_metric import UtilityMetric


@pytest.fixture
def dialogues():
    """Load test dialogues."""
    return json_to_dialogues(
        "tests/data/annotated_dialogues.json",
        agent_ids=["Agent"],
        user_ids=["User"],
    )


FIXED_UTILITY = {
    "success": 1.0,
    "successful_recommendation_round_ratio": 0.5,
    "reward_per_dialogue_length": 0.1,
}


@pytest.fixture
def metric(dialogues):
    """UtilityMetric returning fixed metrics."""
    with patch.object(
        UtilityMetric, "_get_utility_metrics", return_value=FIXED_UTILITY
    ):
        yield UtilityMetric(
            user_nlu_config_path="dummy_user_nlu.yaml",
            agent_nlu_config_path="dummy_agent_nlu.yaml",
        )


def test_evaluate_dialogue(metric: UtilityMetric, dialogues) -> None:
    """Test evaluate_dialogue returns selected metric as float."""
    dialogue = dialogues[0]
    assert metric.evaluate_dialogue(dialogue) == 1.0
    assert metric.evaluate_dialogue(dialogue, metric="success") == 1.0
    assert (
        metric.evaluate_dialogue(
            dialogue, metric="successful_recommendation_round_ratio"
        )
        == 0.5
    )
    assert (
        metric.evaluate_dialogue(dialogue, metric="reward_per_dialogue_length")
        == 0.1
    )


def test_evaluate_dialogues(metric: UtilityMetric, dialogues) -> None:
    """Test evaluate_dialogues returns conversation_id -> full metrics dict."""
    result = metric.evaluate_dialogues(dialogues)
    assert len(result) == len(dialogues)
    for dialogue in dialogues:
        assert dialogue.conversation_id in result
        assert result[dialogue.conversation_id] == FIXED_UTILITY


def test_evaluate_agents(metric: UtilityMetric, dialogues) -> None:
    """Test evaluate_agents returns agent_id -> {conversation_id -> metrics
    dict}."""
    result = metric.evaluate_agents(dialogues)
    assert "Agent" in result
    agent_scores = result["Agent"]
    assert len(agent_scores) == len(dialogues)
    for dialogue in dialogues:
        assert dialogue.conversation_id in agent_scores
        conv_metrics = agent_scores[dialogue.conversation_id]
        assert conv_metrics == FIXED_UTILITY
