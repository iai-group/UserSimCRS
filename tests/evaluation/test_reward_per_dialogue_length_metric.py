"""Tests for the reward-per-dialogue-length metric."""

import pytest

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.intent import Intent
from dialoguekit.participant.participant import DialogueParticipant

from usersimcrs.evaluation.reward_per_dialogue_length_metric import (
    RewardPerDialogueLengthMetric,
)

ACCEPTANCE_INTENTS = [Intent("ACC")]


@pytest.fixture
def reward_per_dialogue_length_metric() -> RewardPerDialogueLengthMetric:
    """Instantiates the reward-per-dialogue-length metric."""
    return RewardPerDialogueLengthMetric()


def test_reward_per_dialogue_length_counts_user_acceptances(
    reward_per_dialogue_length_metric: RewardPerDialogueLengthMetric,
) -> None:
    """Verifies that only user acceptances contribute to the reward."""
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
            text="agent followup",
            participant=DialogueParticipant.AGENT,
            dialogue_acts=[],
        )
    )
    dialogue.add_utterance(
        AnnotatedUtterance(
            text="accept again",
            participant=DialogueParticipant.USER,
            dialogue_acts=[DialogueAct(Intent("ACC"))],
        )
    )

    assert reward_per_dialogue_length_metric.evaluate_dialogue(
        dialogue, ACCEPTANCE_INTENTS
    ) == pytest.approx(2 / 4)
