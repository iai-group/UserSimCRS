"""Tests for utility metric classes."""

from unittest.mock import MagicMock, patch

import pytest

from dialoguekit.utils.dialogue_reader import json_to_dialogues

from usersimcrs.evaluation.reward_per_dialogue_length_metric import (
    RewardPerDialogueLengthMetric,
)
from usersimcrs.evaluation.success_rate_metric import SuccessRateMetric
from usersimcrs.evaluation.successful_recommendation_round_ratio_metric import (
    SuccessfulRecommendationRoundRatioMetric,
)

_MOCK_NLU = MagicMock()


@pytest.fixture
def dialogues():
    """Load test dialogues."""
    return json_to_dialogues(
        "tests/data/annotated_dialogues.json",
        agent_ids=["Agent"],
        user_ids=["User"],
    )


@pytest.fixture
def success_rate_metric():
    return SuccessRateMetric(
        user_nlu_config_path="dummy_user_nlu.yaml",
        agent_nlu_config_path="dummy_agent_nlu.yaml",
    )


@pytest.fixture
def successful_round_ratio_metric():
    return SuccessfulRecommendationRoundRatioMetric(
        user_nlu_config_path="dummy_user_nlu.yaml",
        agent_nlu_config_path="dummy_agent_nlu.yaml",
    )


@pytest.fixture
def reward_per_dialogue_length_metric():
    return RewardPerDialogueLengthMetric(
        user_nlu_config_path="dummy_user_nlu.yaml",
        agent_nlu_config_path="dummy_agent_nlu.yaml",
    )


def test_success_rate_evaluate_dialogue(
    success_rate_metric: SuccessRateMetric, dialogues
) -> None:
    """Test SuccessRateMetric.evaluate_dialogue."""
    dialogue = dialogues[0]
    with (
        patch(
            "usersimcrs.evaluation.success_rate_metric.prepare_dialogue",
            return_value=(dialogue, [], [], [], _MOCK_NLU, _MOCK_NLU),
        ),
        patch.object(SuccessRateMetric, "_assess_dialogue", return_value=True),
    ):
        assert success_rate_metric.evaluate_dialogue(dialogue) == 1.0


def test_success_rate_evaluate_dialogue_unsuccessful(
    success_rate_metric: SuccessRateMetric, dialogues
) -> None:
    """Test SuccessRateMetric.evaluate_dialogue for failed dialogue."""
    dialogue = dialogues[0]
    with (
        patch(
            "usersimcrs.evaluation.success_rate_metric.prepare_dialogue",
            return_value=(dialogue, [], [], [], _MOCK_NLU, _MOCK_NLU),
        ),
        patch.object(SuccessRateMetric, "_assess_dialogue", return_value=False),
    ):
        assert success_rate_metric.evaluate_dialogue(dialogue) == 0.0


def test_successful_recommendation_round_ratio_evaluate_dialogue(
    successful_round_ratio_metric: SuccessfulRecommendationRoundRatioMetric,
    dialogues,
) -> None:
    """Test SuccessfulRecommendationRoundRatioMetric.evaluate_dialogue."""
    dialogue = dialogues[0]
    with (
        patch(
            "usersimcrs.evaluation.successful_recommendation_round_ratio_metric"
            ".prepare_dialogue",
            return_value=(dialogue, [], [], [], _MOCK_NLU, _MOCK_NLU),
        ),
        patch.object(
            SuccessfulRecommendationRoundRatioMetric,
            "_assess_dialogue",
            return_value=(1, 2),
        ),
    ):
        assert successful_round_ratio_metric.evaluate_dialogue(dialogue) == 0.5


def test_reward_per_dialogue_length_evaluate_dialogue(
    reward_per_dialogue_length_metric: RewardPerDialogueLengthMetric, dialogues
) -> None:
    """Test RewardPerDialogueLengthMetric.evaluate_dialogue."""
    dialogue = dialogues[0]
    with (
        patch(
            "usersimcrs.evaluation.reward_per_dialogue_length_metric"
            ".prepare_dialogue",
            return_value=(dialogue, [], [], [], _MOCK_NLU, _MOCK_NLU),
        ),
        patch.object(
            RewardPerDialogueLengthMetric,
            "_assess_dialogue",
            return_value=(1, 10),
        ),
    ):
        assert (
            reward_per_dialogue_length_metric.evaluate_dialogue(dialogue) == 0.1
        )
