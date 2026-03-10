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

_LOAD_NLU_PATH = "usersimcrs.evaluation.utility_base_metric.load_nlu"
_ANNOTATE_PATH = "usersimcrs.evaluation.utility_base_metric.annotate_dialogue"


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


def test_success_rate_init() -> None:
    """Test SuccessRateMetric default and custom name."""
    metric = SuccessRateMetric()
    assert metric.name == "success_rate"

    metric = SuccessRateMetric(name="custom_sr")
    assert metric.name == "custom_sr"


def test_success_rate_evaluate_dialogue(
    success_rate_metric: SuccessRateMetric, dialogues
) -> None:
    """Test SuccessRateMetric returns 1.0 for accepted dialogue."""
    dialogue = dialogues[0]
    mock_nlu = MagicMock()
    with (
        patch(_LOAD_NLU_PATH, return_value=mock_nlu),
        patch(_ANNOTATE_PATH),
        patch.object(
            SuccessRateMetric,
            "_assess_dialogue",
            return_value=True,
        ),
    ):
        assert success_rate_metric.evaluate_dialogue(dialogue) == 1.0


def test_success_rate_without_nlu_paths(dialogues) -> None:
    """Test SuccessRateMetric works on pre-annotated dialogues."""
    metric = SuccessRateMetric()
    dialogue = dialogues[0]
    with patch.object(SuccessRateMetric, "_assess_dialogue", return_value=True):
        assert metric.evaluate_dialogue(dialogue) == 1.0


@pytest.fixture
def srrr_metric():
    return SuccessfulRecommendationRoundRatioMetric()


def test_srrr_init() -> None:
    """Test SRRR metric default and custom name."""
    metric = SuccessfulRecommendationRoundRatioMetric()
    assert metric.name == "successful_recommendation_round_ratio"

    metric = SuccessfulRecommendationRoundRatioMetric(name="srrr_v2")
    assert metric.name == "srrr_v2"


def test_srrr_evaluate_dialogue(srrr_metric, dialogues) -> None:
    """Test SRRR returns correct ratio."""
    dialogue = dialogues[0]
    with patch.object(
        SuccessfulRecommendationRoundRatioMetric,
        "_assess_dialogue",
        return_value=(1, 2),
    ):
        assert srrr_metric.evaluate_dialogue(dialogue) == 0.5


def test_srrr_all_rounds_successful(srrr_metric, dialogues) -> None:
    """Test SRRR returns 1.0 when all rounds accepted."""
    dialogue = dialogues[0]
    with patch.object(
        SuccessfulRecommendationRoundRatioMetric,
        "_assess_dialogue",
        return_value=(3, 3),
    ):
        assert srrr_metric.evaluate_dialogue(dialogue) == 1.0


def test_srrr_no_successful_rounds(srrr_metric, dialogues) -> None:
    """Test SRRR returns 0.0 when no rounds are accepted."""
    dialogue = dialogues[0]
    with patch.object(
        SuccessfulRecommendationRoundRatioMetric,
        "_assess_dialogue",
        return_value=(0, 4),
    ):
        assert srrr_metric.evaluate_dialogue(dialogue) == 0.0


@pytest.fixture
def rdl_metric():
    return RewardPerDialogueLengthMetric()


def test_rdl_no_accepted(rdl_metric, dialogues) -> None:
    """Test RDL returns 0.0 when no recommendations accepted."""
    dialogue = dialogues[0]
    with patch.object(
        RewardPerDialogueLengthMetric,
        "_assess_dialogue",
        return_value=(0, 7),
    ):
        assert rdl_metric.evaluate_dialogue(dialogue) == 0.0


def test_rdl_multiple_accepted(rdl_metric, dialogues) -> None:
    """Test RDL with several accepted recommendations."""
    dialogue = dialogues[0]
    with patch.object(
        RewardPerDialogueLengthMetric,
        "_assess_dialogue",
        return_value=(3, 10),
    ):
        assert rdl_metric.evaluate_dialogue(dialogue) == pytest.approx(0.3)
