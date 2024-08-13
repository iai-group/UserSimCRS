"""Tests for the InteractionModel class."""

import pytest

from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.slot_value_annotation import SlotValueAnnotation
from dialoguekit.utils.dialogue_reader import json_to_dialogues
from usersimcrs.core.information_need import InformationNeed
from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.simulator.agenda_based.interaction_model import (
    InteractionModel,
)

ANNOTATED_CONVERSATIONS = json_to_dialogues(
    "tests/data/annotated_dialogues.json",
    agent_ids=["Agent"],
    user_ids=["User"],
)


@pytest.fixture
def im_crsv1(
    information_need: InformationNeed, domain: SimulationDomain
) -> InteractionModel:
    """CRS v1 Interaction model fixture."""
    im = InteractionModel(
        "tests/data/interaction_models/crs_v1.yaml",
        domain,
        ANNOTATED_CONVERSATIONS,
    )
    im.initialize_agenda(information_need)
    return im


def test_initialize_agenda(
    im_crsv1: InteractionModel, information_need: InformationNeed
) -> None:
    im_crsv1.initialize_agenda(information_need)
    assert len(im_crsv1.agenda.stack) == 6
    assert im_crsv1.agenda.stack[0].intent == im_crsv1.INTENT_START
    assert im_crsv1.agenda.stack[1].intent == im_crsv1.INTENT_DISCLOSE
    assert im_crsv1.agenda.stack[-2] == DialogueAct(
        im_crsv1.INTENT_INQUIRE, [SlotValueAnnotation("RATING")]
    )
    assert im_crsv1.agenda.stack[-1].intent == im_crsv1.INTENT_STOP


def test_get_next_dialogue_acts(im_crsv1: InteractionModel) -> None:
    dialogue_acts = im_crsv1.get_next_dialogue_acts(3)
    assert len(dialogue_acts) == 3
    assert dialogue_acts[0].intent == im_crsv1.INTENT_START
    assert dialogue_acts[1].intent == im_crsv1.INTENT_DISCLOSE
    assert dialogue_acts[2] == DialogueAct(
        im_crsv1.INTENT_DISCLOSE,
        [SlotValueAnnotation("DIRECTOR", "Steven Spielberg")],
    )
