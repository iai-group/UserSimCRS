"""Tests for the success rate metric."""

import pytest

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.intent import Intent
from dialoguekit.participant.participant import DialogueParticipant

from usersimcrs.evaluation.success_rate_metric import SuccessRateMetric

RECOMMENDATION_INTENTS = [Intent("REC-S"), Intent("REC-E")]
ACCEPTANCE_INTENTS = [Intent("ACC")]
REJECTION_INTENTS = [Intent("REJ")]


@pytest.fixture
def success_rate_metric() -> SuccessRateMetric:
    """Instantiates the success rate metric."""
    return SuccessRateMetric()


def test_success_rate_metric_single_accepted_round(
    success_rate_metric: SuccessRateMetric,
) -> None:
    """Verifies that an accepted recommendation yields a success rate of 1."""
    dialogue = Dialogue(agent_id="Agent", user_id="User", conversation_id="cid")
    dialogue.add_utterance(
        AnnotatedUtterance(
            text="rec",
            participant=DialogueParticipant.AGENT,
            dialogue_acts=[DialogueAct(Intent("REC-S"))],
        )
    )
    dialogue.add_utterance(
        AnnotatedUtterance(
            text="accept",
            participant=DialogueParticipant.USER,
            dialogue_acts=[DialogueAct(Intent("ACC"))],
        )
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
    """Verifies that a rejection in the round yields a success rate of 0."""
    dialogue = Dialogue(agent_id="Agent", user_id="User", conversation_id="cid")
    dialogue.add_utterance(
        AnnotatedUtterance(
            text="rec",
            participant=DialogueParticipant.AGENT,
            dialogue_acts=[DialogueAct(Intent("REC-S"))],
        )
    )
    dialogue.add_utterance(
        AnnotatedUtterance(
            text="accept",
            participant=DialogueParticipant.USER,
            dialogue_acts=[DialogueAct(Intent("ACC"))],
        )
    )
    dialogue.add_utterance(
        AnnotatedUtterance(
            text="no",
            participant=DialogueParticipant.USER,
            dialogue_acts=[DialogueAct(Intent("REJ"))],
        )
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
