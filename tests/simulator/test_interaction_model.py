"""Tests for the InteractionModel class."""

import pytest

from cryses.simulator.interaction_model import InteractionModel


# List of user intents in agenda
ANNOTATED_CONVERSATIONS = [
        {
            "conversation ID": "1",
            "conversation": [
                {
                    "participant": "USER",
                    "intent": "DISCLOSE.NON-DISCLOSE"
                },
                {
                    "participant": "AGENT",
                    "intent": "INQUIRE"
                },
                {
                    "participant": "USER",
                    "intent": "Intent-A"
                },
                {
                    "participant": "AGENT",
                    "intent": "INQUIRE"
                },
                {
                    "participant": "USER",
                    "intent": "Intent-B"
                },
                {
                    "participant": "AGENT",
                    "intent": "INQUIRE"
                },
                {
                    "participant": "USER",
                    "intent": "Intent-C"
                }
            ]
        },
        {
            "conversation ID": "2",
            "conversation": [
                {
                    "participant": "USER",
                    "intent": "DISCLOSE.NON-DISCLOSE"
                },
                {
                    "participant": "AGENT",
                    "intent": "INQUIRE"
                },
                {
                    "participant": "USER",
                    "intent": "Intent-A"
                },
                {
                    "participant": "AGENT",
                    "intent": "INQUIRE"
                },
                {
                    "participant": "USER",
                    "intent": "Intent-A"
                },
                {
                    "participant": "AGENT",
                    "intent": "INQUIRE"
                },
                {
                    "participant": "USER",
                    "intent": "Intent-C"
                }
            ]
        },
        {
            "conversation ID": "3",
            "conversation": [
                {
                    "participant": "USER",
                    "intent": "DISCLOSE.NON-DISCLOSE"
                },
                {
                    "participant": "AGENT",
                    "intent": "INQUIRE"
                },
                {
                    "participant": "USER",
                    "intent": "Intent-A"
                },
                {
                    "participant": "AGENT",
                    "intent": "INQUIRE"
                },
                {
                    "participant": "USER",
                    "intent": "Intent-C"
                },
                {
                    "participant": "AGENT",
                    "intent": "INQUIRE"
                },
                {
                    "participant": "USER",
                    "intent": "Intent-C"
                }
            ]
        }
    ]


# CIR6 Interaction model.
@pytest.fixture
def im_cir6():
    return InteractionModel("data/interaction_models/cir6.yaml", ANNOTATED_CONVERSATIONS)


def test_is_agent_intent_elicit(im_cir6):
    assert im_cir6.is_agent_intent_elicit("INQUIRE")
    assert not im_cir6.is_agent_intent_elicit("REVEAL")


def test_is_agent_intent_set_retrieval(im_cir6):
    assert im_cir6.is_agent_intent_set_retrieval("REVEAL")
    assert not im_cir6.is_agent_intent_set_retrieval("INQUIRE")


def test_intent_distribution(im_cir6):
    user_intent_distribution, intent_distribution = im_cir6.intent_distribution(
        ANNOTATED_CONVERSATIONS
    )
    expected_user_intent_distribution = {
        "DISCLOSE.NON-DISCLOSE": {"Intent-A": 3},
        "Intent-A": {"Intent-B": 1, "Intent-A": 1, "Intent-C": 2},
        "Intent-B": {"Intent-C": 1},
        "Intent-C": {"STOP": 3, "Intent-C": 1}
    }
    expected_intent_distribution = {
        "INQUIRE": {"Intent-A": 4, "Intent-B": 1, "Intent-C": 4}
    }
    assert user_intent_distribution == expected_user_intent_distribution
    assert intent_distribution == expected_intent_distribution


def test_next_intent(im_cir6):
    user_intent_distribution, intent_distribution = im_cir6.intent_distribution(
        ANNOTATED_CONVERSATIONS
    )
    assert im_cir6.next_intent("DISCLOSE.NON-DISCLOSE", user_intent_distribution)
    # Note the next intent is picked randomly, so we only check its eixistence.
    assert im_cir6.next_intent("Intent-A", user_intent_distribution)
    assert im_cir6.next_intent("Intent-C", user_intent_distribution)
    assert im_cir6.next_intent("Intent-B", user_intent_distribution) == "Intent-C"


def test_initialize_agenda(im_cir6):
    assert len(im_cir6.initialize_agenda()) > 0
    assert len(im_cir6.agenda) > 0
    assert im_cir6.agenda[-1] == "DISCLOSE.NON-DISCLOSE"


def test_update_agenda(im_cir6):
    next_intent = im_cir6.update_agenda("INQUIRE")
    assert next_intent
