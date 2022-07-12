"""Tests for the InteractionModel class."""

import pytest

from usersimcrs.simulator.interaction_model import InteractionModel
from dialoguekit.core.intent import Intent

# List of user intents in agenda
ANNOTATED_CONVERSATIONS = [
    {
        "conversation ID": "1",
        "conversation": [
            {"participant": "USER", "intent": "DISCLOSE.NON-DISCLOSE"},
            {"participant": "AGENT", "intent": "INQUIRE"},
            {"participant": "USER", "intent": "Intent-A"},
            {"participant": "AGENT", "intent": "INQUIRE"},
            {"participant": "USER", "intent": "Intent-B"},
            {"participant": "AGENT", "intent": "INQUIRE"},
            {"participant": "USER", "intent": "Intent-C"},
        ],
    },
    {
        "conversation ID": "2",
        "conversation": [
            {"participant": "USER", "intent": "DISCLOSE.NON-DISCLOSE"},
            {"participant": "AGENT", "intent": "INQUIRE"},
            {"participant": "USER", "intent": "Intent-A"},
            {"participant": "AGENT", "intent": "INQUIRE"},
            {"participant": "USER", "intent": "Intent-A"},
            {"participant": "AGENT", "intent": "INQUIRE"},
            {"participant": "USER", "intent": "Intent-C"},
        ],
    },
    {
        "conversation ID": "3",
        "conversation": [
            {"participant": "USER", "intent": "DISCLOSE.NON-DISCLOSE"},
            {"participant": "AGENT", "intent": "INQUIRE"},
            {"participant": "USER", "intent": "Intent-A"},
            {"participant": "AGENT", "intent": "INQUIRE"},
            {"participant": "USER", "intent": "Intent-C"},
            {"participant": "AGENT", "intent": "INQUIRE"},
            {"participant": "USER", "intent": "Intent-C"},
        ],
    },
]


_INTENT_INQUIRE = Intent("INQUIRE")
_INTENT_REVEAL = Intent("REVEAL")
_INTENT_DISCLOSE_NON_DISCLOSE = Intent("DISCLOSE.NON-DISCLOSE")
_INTENT_A = Intent("Intent-A")
_INTENT_B = Intent("Intent-B")
_INTENT_C = Intent("Intent-C")
_INTENT_STOP = Intent("STOP")


# CIR6 Interaction model.
@pytest.fixture
def im_cir6():
    return InteractionModel(
        "data/interaction_models/crs_v1.yaml", ANNOTATED_CONVERSATIONS
    )


def test_is_agent_intent_elicit(im_cir6):
    assert im_cir6.is_agent_intent_elicit(_INTENT_INQUIRE)
    assert not im_cir6.is_agent_intent_elicit(_INTENT_REVEAL)


def test_is_agent_intent_set_retrieval(im_cir6):
    assert im_cir6.is_agent_intent_set_retrieval(_INTENT_REVEAL)
    assert not im_cir6.is_agent_intent_set_retrieval(_INTENT_INQUIRE)


def test_intent_distribution(im_cir6):
    user_intent_distribution, intent_distribution = im_cir6.intent_distribution(
        ANNOTATED_CONVERSATIONS
    )
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


def test_next_intent(im_cir6):
    user_intent_distribution, intent_distribution = im_cir6.intent_distribution(
        ANNOTATED_CONVERSATIONS
    )
    assert im_cir6.next_intent(
        _INTENT_DISCLOSE_NON_DISCLOSE, user_intent_distribution
    )
    # Note the next intent is picked randomly, so we only check its eixistence.
    assert im_cir6.next_intent(_INTENT_A, user_intent_distribution)
    assert im_cir6.next_intent(_INTENT_C, user_intent_distribution)
    assert im_cir6.next_intent(_INTENT_B, user_intent_distribution) == _INTENT_C


def test_initialize_agenda(im_cir6):
    assert len(im_cir6.initialize_agenda()) > 0
    assert len(im_cir6.agenda) > 0
    assert im_cir6.agenda[-1] == _INTENT_DISCLOSE_NON_DISCLOSE


def test_update_agenda(im_cir6):
    next_intent = im_cir6.update_agenda(_INTENT_INQUIRE)
    assert next_intent
