"""Tests for the InteractionModel class."""

import pytest

from cryses.simulator.interaction_model import InteractionModel


# List of user intents in agenda
@pytest.fixture
def agenda_list():
    return [
        [
            ("USER", "DISCLOSE.NON-DISCLOSE"),
            ("AGENT", "INQUIRE"),
            ("USER", "Intent-A"),
            ("AGENT", "INQUIRE"),
            ("USER", "Intent-B"),
            ("AGENT", "INQUIRE"),
            ("USER", "Intent-C"),
        ],
        [
            ("USER", "DISCLOSE.NON-DISCLOSE"),
            ("AGENT", "INQUIRE"),
            ("USER", "Intent-A"),
            ("AGENT", "INQUIRE"),
            ("USER", "Intent-A"),
            ("AGENT", "INQUIRE"),
            ("USER", "Intent-C"),
        ],
        [
            ("USER", "DISCLOSE.NON-DISCLOSE"),
            ("AGENT", "INQUIRE"),
            ("USER", "Intent-A"),
            ("AGENT", "INQUIRE"),
            ("USER", "Intent-C"),
            ("AGENT", "INQUIRE"),
            ("USER", "Intent-C"),
        ],
    ]


# CIR6 Interaction model.
@pytest.fixture
def im_cir6(agenda_list):
    return InteractionModel("../../data/interaction_models/cir6.yaml", agenda_list)


def test_is_agent_intent_elicit(im_cir6):
    assert im_cir6.is_agent_intent_elicit("INQUIRE")
    assert not im_cir6.is_agent_intent_elicit("REVEAL")


def test_is_agent_intent_set_retrieval(im_cir6):
    assert im_cir6.is_agent_intent_set_retrieval("REVEAL")
    assert not im_cir6.is_agent_intent_set_retrieval("INQUIRE")


def test_intent_distribution(im_cir6, agenda_list):
    user_intent_distribution, intent_distribution = im_cir6.intent_distribution(
        agenda_list
    )
    exepcted_user_intent_distribution = {
        "DISCLOSE.NON-DISCLOSE": {"Intent-A": 3},
        "Intent-A": {"Intent-B": 1, "Intent-A": 1, "Intent-C": 2},
        "Intent-B": {"Intent-C": 1},
        "Intent-C": {"STOP": 3, "Intent-C": 1},
    }
    exepcted_intent_distribution = {
        "INQUIRE": {"Intent-A": 4, "Intent-B": 1, "Intent-C": 4}
    }
    assert user_intent_distribution == exepcted_user_intent_distribution
    assert intent_distribution == exepcted_intent_distribution


def test_next_intent(im_cir6, agenda_list):
    user_intent_distribution, intent_distribution = im_cir6.intent_distribution(
        agenda_list
    )
    assert im_cir6.next_intent("DISCLOSE.NON-DISCLOSE", user_intent_distribution)
    assert im_cir6.next_intent("Intent-A", user_intent_distribution)
    assert im_cir6.next_intent("Intent-C", user_intent_distribution)
    assert im_cir6.next_intent("Intent-B", user_intent_distribution) == "Intent-C"


def test_initialize_agenda(im_cir6):
    assert len(im_cir6.initialize_agenda()) > 0
    assert len(im_cir6.agenda) > 0
    assert im_cir6.agenda[-1] == "DISCLOSE.NON-DISCLOSE"


def test_update_agenda(im_cir6):
    agenda_intent = "INQUIRE"
    next_intent = im_cir6.update_agenda(agenda_intent)
    assert next_intent
