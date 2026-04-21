"""Tests for the success rate metric."""

from typing import List

import pytest

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.intent import Intent
from dialoguekit.participant.participant import DialogueParticipant

from usersimcrs.evaluation.success_rate_metric import SuccessRateMetric


@pytest.fixture
def success_rate_metric() -> SuccessRateMetric:
    """Instantiates the success rate metric."""
    return SuccessRateMetric()


def test_success_rate_metric_single_accepted_round(
    success_rate_metric: SuccessRateMetric,
    recommendation_intents: List[Intent],
    acceptance_intents: List[Intent],
    rejection_intents: List[Intent],
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
            recommendation_intents,
            acceptance_intents,
            rejection_intents,
        )
        == 1.0
    )


def test_success_rate_metric_accepted_round_with_rejection(
    success_rate_metric: SuccessRateMetric,
    recommendation_intents: List[Intent],
    acceptance_intents: List[Intent],
    rejection_intents: List[Intent],
) -> None:
    """Verifies that an accepted recommendation yields a success rate of 0."""
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
            recommendation_intents,
            acceptance_intents,
            rejection_intents,
        )
        == 0.0
    )


def test_success_rate_metric_accepted_round_with_acceptance(
    success_rate_metric: SuccessRateMetric,
    recommendation_intents: List[Intent],
    acceptance_intents: List[Intent],
    rejection_intents: List[Intent],
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
            text="no",
            participant=DialogueParticipant.USER,
            dialogue_acts=[DialogueAct(Intent("REJ"))],
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
            recommendation_intents,
            acceptance_intents,
            rejection_intents,
        )
        == 1.0
    )


def test_success_rate_metric_accepted_2_rounds(
    success_rate_metric: SuccessRateMetric,
    recommendation_intents: List[Intent],
    acceptance_intents: List[Intent],
    rejection_intents: List[Intent],
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
            text="no",
            participant=DialogueParticipant.USER,
            dialogue_acts=[DialogueAct(Intent("REJ"))],
        )
    )
    dialogue.add_utterance(
        AnnotatedUtterance(
            text="accept",
            participant=DialogueParticipant.USER,
            dialogue_acts=[DialogueAct(Intent("ACC"))],
        )
    )
    # 2 round
    dialogue.add_utterance(
        AnnotatedUtterance(
            text="rec2",
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
            recommendation_intents,
            acceptance_intents,
            rejection_intents,
        )
        == 1.0
    )


def test_success_rate_metric_rejected_round(
    success_rate_metric: SuccessRateMetric,
    recommendation_intents: List[Intent],
    acceptance_intents: List[Intent],
    rejection_intents: List[Intent],
) -> None:
    """Verifies that a rejected recommendation yields a success rate of 0."""
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
            text="no",
            participant=DialogueParticipant.USER,
            dialogue_acts=[DialogueAct(Intent("REJ"))],
        )
    )

    assert (
        success_rate_metric.evaluate_dialogue(
            dialogue,
            recommendation_intents,
            acceptance_intents,
            rejection_intents,
        )
        == 0.0
    )
