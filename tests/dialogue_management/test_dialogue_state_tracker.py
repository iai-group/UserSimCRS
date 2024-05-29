"""Tests for the dialogue state tracker module."""

import pytest
from dialoguekit.core.annotation import Annotation
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.intent import Intent
from dialoguekit.participant import DialogueParticipant

from usersimcrs.dialogue_management.dialogue_state_tracker import (
    DialogueStateTracker,
)


@pytest.fixture(scope="module")
def dialogue_state_tracker() -> DialogueStateTracker:
    """Fixture for the dialogue state tracker."""
    dst = DialogueStateTracker()

    initial_state = dst.get_current_state()
    assert initial_state.turn_count == 0
    assert initial_state.agent_dacts == []
    assert initial_state.user_dacts == []
    assert initial_state.belief_state == {}
    return dst


def test_update_state_agent(
    dialogue_state_tracker: DialogueStateTracker,
) -> None:
    """Tests dialogue state update with agent dialogue acts."""
    dialogue_acts = [
        DialogueAct(Intent("greets")),
        DialogueAct(Intent("elicit"), annotations=[Annotation("GENRE")]),
    ]

    dialogue_state_tracker.update_state(
        dialogue_acts, DialogueParticipant.AGENT
    )

    current_state = dialogue_state_tracker.get_current_state()
    assert current_state.turn_count == 1
    assert current_state.agent_dacts == [dialogue_acts]
    assert current_state.user_dacts == []
    assert current_state.belief_state == {"GENRE": None}


def test_update_state_user(
    dialogue_state_tracker: DialogueStateTracker,
) -> None:
    """Tests dialogue state update with user dialogue acts."""
    dialogue_acts = [
        DialogueAct(
            Intent("inform"), annotations=[Annotation("GENRE", "comedy")]
        ),
        DialogueAct(Intent("request"), annotations=[Annotation("YEAR")]),
    ]

    dialogue_state_tracker.update_state(dialogue_acts, DialogueParticipant.USER)

    current_state = dialogue_state_tracker.get_current_state()
    assert current_state.turn_count == 2
    assert len(current_state.agent_dacts) == 1
    assert current_state.user_dacts == [dialogue_acts]
    assert current_state.belief_state == {"GENRE": "comedy", "YEAR": None}
