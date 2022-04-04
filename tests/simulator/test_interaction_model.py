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


# CIR6 Interaction model.
@pytest.fixture
def im_cir6():
    return InteractionModel(
        "data/interaction_models/cir6.yaml", ANNOTATED_CONVERSATIONS
    )


def test_is_agent_intent_elicit(im_cir6):
    assert im_cir6.is_agent_intent_elicit(Intent("INQUIRE"))
    assert not im_cir6.is_agent_intent_elicit(Intent("REVEAL"))


def test_is_agent_intent_set_retrieval(im_cir6):
    assert im_cir6.is_agent_intent_set_retrieval(Intent("REVEAL"))
    assert not im_cir6.is_agent_intent_set_retrieval(Intent("INQUIRE"))


def test_intent_distribution(im_cir6):
    user_intent_distribution, intent_distribution = im_cir6.intent_distribution(
        ANNOTATED_CONVERSATIONS
    )
    expected_user_intent_distribution = {
        "DISCLOSE.NON-DISCLOSE": {Intent("Intent-A").label: 3},
        "Intent-A": {
            Intent("Intent-B").label: 1,
            Intent("Intent-A").label: 1,
            Intent("Intent-C").label: 2,
        },
        "Intent-B": {Intent("Intent-C").label: 1},
        "Intent-C": {Intent("STOP").label: 3, Intent("Intent-C").label: 1},
    }
    expected_intent_distribution = {
        "INQUIRE": {
            Intent("Intent-A").label: 4,
            Intent("Intent-B").label: 1,
            Intent("Intent-C").label: 4,
        }
    }
    assert user_intent_distribution == expected_user_intent_distribution
    assert intent_distribution == expected_intent_distribution


def test_next_intent(im_cir6):
    user_intent_distribution, intent_distribution = im_cir6.intent_distribution(
        ANNOTATED_CONVERSATIONS
    )
    assert im_cir6.next_intent(
        Intent("DISCLOSE.NON-DISCLOSE"), user_intent_distribution
    )
    # Note the next intent is picked randomly, so we only check its eixistence.
    assert im_cir6.next_intent(Intent("Intent-A"), user_intent_distribution)
    assert im_cir6.next_intent(Intent("Intent-C"), user_intent_distribution)
    assert (
        im_cir6.next_intent(Intent("Intent-B"), user_intent_distribution).label
        == "Intent-C"
    )


def test_initialize_agenda(im_cir6):
    assert len(im_cir6.initialize_agenda()) > 0
    assert len(im_cir6.agenda) > 0
    assert im_cir6.agenda[-1].label == "DISCLOSE.NON-DISCLOSE"


def test_update_agenda(im_cir6):
    next_intent = im_cir6.update_agenda(Intent("INQUIRE"))
    assert next_intent
