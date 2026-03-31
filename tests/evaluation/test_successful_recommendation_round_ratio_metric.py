"""Tests for the successful recommendation round ratio metric."""

import pytest

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.intent import Intent
from dialoguekit.participant.participant import DialogueParticipant

from usersimcrs.evaluation.successful_recommendation_round_ratio_metric import (
    SuccessfulRecommendationRoundRatioMetric,
)

RECOMMENDATION_INTENTS = [Intent("REC-S"), Intent("REC-E")]
ACCEPTANCE_INTENTS = [Intent("ACC")]
REJECTION_INTENTS = [Intent("REJ")]


@pytest.fixture
def successful_round_ratio_metric() -> (
    SuccessfulRecommendationRoundRatioMetric
):
    """Instantiates the successful recommendation round ratio metric."""
    return SuccessfulRecommendationRoundRatioMetric()


def test_successful_round_ratio_handles_no_recommendations(
    successful_round_ratio_metric: SuccessfulRecommendationRoundRatioMetric,
) -> None:
    """Verifies that dialogues without recommendations return 0.0."""
    dialogue = Dialogue(agent_id="Agent", user_id="User", conversation_id="cid")
    dialogue.add_utterance(
        AnnotatedUtterance(
            text="hi",
            participant=DialogueParticipant.USER,
            dialogue_acts=[],
        )
    )
    dialogue.add_utterance(
        AnnotatedUtterance(
            text="hello",
            participant=DialogueParticipant.AGENT,
            dialogue_acts=[],
        )
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
    """Verifies that the metric counts successful rounds correctly."""
    dialogue = Dialogue(agent_id="Agent", user_id="User", conversation_id="cid")
    dialogue.add_utterance(
        AnnotatedUtterance(
            text="rec1",
            participant=DialogueParticipant.AGENT,
            dialogue_acts=[DialogueAct(Intent("REC-S"))],
        )
    )
    dialogue.add_utterance(
        AnnotatedUtterance(
            text="accept1",
            participant=DialogueParticipant.USER,
            dialogue_acts=[DialogueAct(Intent("ACC"))],
        )
    )
    dialogue.add_utterance(
        AnnotatedUtterance(
            text="rec2",
            participant=DialogueParticipant.AGENT,
            dialogue_acts=[DialogueAct(Intent("REC-E"))],
        )
    )
    dialogue.add_utterance(
        AnnotatedUtterance(
            text="reject2",
            participant=DialogueParticipant.USER,
            dialogue_acts=[DialogueAct(Intent("REJ"))],
        )
    )

    assert successful_round_ratio_metric.evaluate_dialogue(
        dialogue,
        RECOMMENDATION_INTENTS,
        ACCEPTANCE_INTENTS,
        REJECTION_INTENTS,
    ) == pytest.approx(1 / 2)
