"""Tests for utility-centric evaluation metrics.

These tests cover:
- success rate: whether any recommendation round is accepted
- reward per dialogue length: accepted recommendations normalized by dialogue
  length
- successful recommendation round ratio: accepted recommendation rounds
  normalized by total recommendation rounds

The metrics operate purely on annotated dialogue acts (REC/ACC/REJ) and should
be testable without any external NLU or LLM dependencies.
"""

from __future__ import annotations

from typing import Iterable, List

import pytest

# Import order matters for dialoguekit due to circular imports in the library.
from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.intent import Intent
from dialoguekit.participant.participant import DialogueParticipant

from usersimcrs.evaluation.reward_per_dialogue_length_metric import (
    RewardPerDialogueLengthMetric,
)
from usersimcrs.evaluation.success_rate_metric import SuccessRateMetric
from usersimcrs.evaluation.successful_recommendation_round_ratio_metric import (
    SuccessfulRecommendationRoundRatioMetric,
)

RECOMMENDATION_INTENTS = [Intent("REC-S"), Intent("REC-E")]
ACCEPTANCE_INTENTS = [Intent("ACC")]
REJECTION_INTENTS = [Intent("REJ")]


def _annotated_utterance(
    text: str, participant: DialogueParticipant, intents: Iterable[str]
) -> AnnotatedUtterance:
    intent_objects = [Intent(i) for i in intents]
    dialogue_acts = [DialogueAct(intent) for intent in intent_objects]

    try:
        utterance = AnnotatedUtterance(
            text=text,
            participant=participant,
            dialogue_acts=dialogue_acts,
        )
    except TypeError:
        try:
            utterance = AnnotatedUtterance(
                text=text,
                participant=participant,
                intent=intent_objects[0] if intent_objects else None,
            )
        except TypeError:
            utterance = AnnotatedUtterance(
                text=text,
                participant=participant,
            )

    utterance.dialogue_acts = dialogue_acts
    utterance.get_intents = lambda: intent_objects
    return utterance


def _dialogue(
    conversation_id: str, utterances: List[AnnotatedUtterance]
) -> Dialogue:
    d = Dialogue(
        agent_id="Agent", user_id="User", conversation_id=conversation_id
    )
    for u in utterances:
        d.add_utterance(u)
    return d


@pytest.fixture
def success_rate_metric() -> SuccessRateMetric:
    """Instantiate the success rate metric."""
    return SuccessRateMetric()


@pytest.fixture
def reward_per_dialogue_length_metric() -> RewardPerDialogueLengthMetric:
    """Instantiate the reward-per-dialogue-length metric."""
    return RewardPerDialogueLengthMetric()


@pytest.fixture
def successful_round_ratio_metric() -> (
    SuccessfulRecommendationRoundRatioMetric
):
    """Instantiate the successful recommendation round ratio metric."""
    return SuccessfulRecommendationRoundRatioMetric()


def test_success_rate_metric_single_accepted_round(
    success_rate_metric: SuccessRateMetric,
) -> None:
    dialogue = _dialogue(
        "cid",
        [
            _annotated_utterance("rec", DialogueParticipant.AGENT, ["REC-S"]),
            _annotated_utterance("accept", DialogueParticipant.USER, ["ACC"]),
        ],
    )
    assert (
        success_rate_metric.evaluate_dialogue(
            dialogue,
            RECOMMENDATION_INTENTS,
            ACCEPTANCE_INTENTS,
            REJECTION_INTENTS,
        )
        == 1.0
    )


def test_success_rate_metric_rejection_overrides_acceptance(
    success_rate_metric: SuccessRateMetric,
) -> None:
    dialogue = _dialogue(
        "cid",
        [
            _annotated_utterance("rec", DialogueParticipant.AGENT, ["REC-S"]),
            _annotated_utterance("accept", DialogueParticipant.USER, ["ACC"]),
            _annotated_utterance("no", DialogueParticipant.USER, ["REJ"]),
        ],
    )
    assert (
        success_rate_metric.evaluate_dialogue(
            dialogue,
            RECOMMENDATION_INTENTS,
            ACCEPTANCE_INTENTS,
            REJECTION_INTENTS,
        )
        == 0.0
    )


def test_reward_per_dialogue_length_counts_user_acceptances(
    reward_per_dialogue_length_metric: RewardPerDialogueLengthMetric,
) -> None:
    dialogue = _dialogue(
        "cid",
        [
            _annotated_utterance("rec", DialogueParticipant.AGENT, ["REC-S"]),
            _annotated_utterance("accept", DialogueParticipant.USER, ["ACC"]),
            _annotated_utterance(
                "agent followup", DialogueParticipant.AGENT, []
            ),
            _annotated_utterance(
                "accept again", DialogueParticipant.USER, ["ACC"]
            ),
        ],
    )
    assert reward_per_dialogue_length_metric.evaluate_dialogue(
        dialogue, ACCEPTANCE_INTENTS
    ) == pytest.approx(2 / 4)


def test_successful_round_ratio_handles_no_recommendations(
    successful_round_ratio_metric: SuccessfulRecommendationRoundRatioMetric,
) -> None:
    dialogue = _dialogue(
        "cid",
        [
            _annotated_utterance("hi", DialogueParticipant.USER, []),
            _annotated_utterance("hello", DialogueParticipant.AGENT, []),
        ],
    )
    assert (
        successful_round_ratio_metric.evaluate_dialogue(
            dialogue,
            RECOMMENDATION_INTENTS,
            ACCEPTANCE_INTENTS,
            REJECTION_INTENTS,
        )
        == 0.0
    )


def test_successful_round_ratio_two_rounds_one_success(
    successful_round_ratio_metric: SuccessfulRecommendationRoundRatioMetric,
) -> None:
    dialogue = _dialogue(
        "cid",
        [
            _annotated_utterance("rec1", DialogueParticipant.AGENT, ["REC-S"]),
            _annotated_utterance("accept1", DialogueParticipant.USER, ["ACC"]),
            _annotated_utterance("rec2", DialogueParticipant.AGENT, ["REC-E"]),
            _annotated_utterance("reject2", DialogueParticipant.USER, ["REJ"]),
        ],
    )
    assert successful_round_ratio_metric.evaluate_dialogue(
        dialogue,
        RECOMMENDATION_INTENTS,
        ACCEPTANCE_INTENTS,
        REJECTION_INTENTS,
    ) == pytest.approx(1 / 2)
