"""Tests for the InteractionModel class."""

import pytest
from dialoguekit.core.intent import Intent
from dialoguekit.utils.dialogue_reader import json_to_dialogues

from usersimcrs.simulator.agenda_based.interaction_model import (
    InteractionModel,
)

ANNOTATED_CONVERSATIONS = json_to_dialogues(
    "tests/data/annotated_dialogues.json",
    agent_ids=["Agent"],
    user_ids=["User"],
)

_INTENT_INQUIRE = Intent("INQUIRE")
_INTENT_REVEAL = Intent("REVEAL")
_INTENT_DISCLOSE_NON_DISCLOSE = Intent("DISCLOSE.NON-DISCLOSE")
_INTENT_A = Intent("Intent-A")
_INTENT_B = Intent("Intent-B")
_INTENT_C = Intent("Intent-C")
_INTENT_STOP = Intent("STOP")


@pytest.fixture
def im_crsv1() -> InteractionModel:
    """CRS v1 Interaction model fixture."""
    return InteractionModel(
        "tests/data/interaction_models/crs_v1.yaml", ANNOTATED_CONVERSATIONS
    )


def test_is_agent_intent_elicit(im_crsv1: InteractionModel) -> None:
    assert im_crsv1.is_agent_intent_elicit(_INTENT_INQUIRE)
    assert not im_crsv1.is_agent_intent_elicit(_INTENT_REVEAL)


def test_is_agent_intent_set_retrieval(im_crsv1: InteractionModel) -> None:
    assert im_crsv1.is_agent_intent_set_retrieval(_INTENT_REVEAL)
    assert not im_crsv1.is_agent_intent_set_retrieval(_INTENT_INQUIRE)


def test_intent_distribution(im_crsv1: InteractionModel) -> None:
    (
        user_intent_distribution,
        intent_distribution,
    ) = im_crsv1.intent_distribution(ANNOTATED_CONVERSATIONS)
    expected_user_intent_distribution = {
        _INTENT_DISCLOSE_NON_DISCLOSE: {_INTENT_A: 3},
        _INTENT_A: {
            _INTENT_B: 1,
            _INTENT_A: 1,
            _INTENT_C: 2,
        },
        _INTENT_B: {_INTENT_C: 1},
        _INTENT_C: {_INTENT_STOP: 3, _INTENT_C: 1},
    }
    expected_intent_distribution = {
        _INTENT_INQUIRE: {
            _INTENT_A: 4,
            _INTENT_B: 1,
            _INTENT_C: 4,
        }
    }
    assert user_intent_distribution == expected_user_intent_distribution
    assert intent_distribution == expected_intent_distribution


def test_next_intent(im_crsv1: InteractionModel) -> None:
    (
        user_intent_distribution,
        _,
    ) = im_crsv1.intent_distribution(ANNOTATED_CONVERSATIONS)
    assert im_crsv1.next_intent(
        _INTENT_DISCLOSE_NON_DISCLOSE, user_intent_distribution
    )
    # Note the next intent is picked randomly, so we only check its existence.
    assert im_crsv1.next_intent(_INTENT_A, user_intent_distribution)
    assert im_crsv1.next_intent(_INTENT_C, user_intent_distribution)
    assert (
        im_crsv1.next_intent(_INTENT_B, user_intent_distribution) == _INTENT_C
    )


def test_initialize_agenda(im_crsv1: InteractionModel) -> None:
    assert len(im_crsv1.initialize_agenda()) > 0
    assert len(im_crsv1.agenda) > 0
    assert im_crsv1.agenda[-1] == _INTENT_DISCLOSE_NON_DISCLOSE


def test_update_agenda(im_crsv1: InteractionModel) -> None:
    initial_intent = im_crsv1.current_intent
    im_crsv1.update_agenda(_INTENT_INQUIRE)
    assert im_crsv1.current_intent is not im_crsv1.INTENT_START
    assert initial_intent is not im_crsv1.current_intent
