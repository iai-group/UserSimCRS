"""Tests for the InteractionModel class."""

import pytest

from cryses.simulator.interaction_model import InteractionModel


# CIR6 Interaction model.
@pytest.fixture
def im_cir6():
    return InteractionModel("data/interaction_models/cir6.yaml")


def test_is_agent_intent_elicit(im_cir6):
    assert im_cir6.is_agent_intent_elicit("INQUIRE")
    assert not im_cir6.is_agent_intent_elicit("REVEAL")


def test_is_agent_intent_set_retrieval(im_cir6):
    assert im_cir6.is_agent_intent_set_retrieval("REVEAL")
    assert not im_cir6.is_agent_intent_set_retrieval("INQUIRE")
