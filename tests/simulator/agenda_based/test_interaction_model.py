"""Tests for the InteractionModel class."""

import pytest

from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.intent import Intent
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


def test_initialize_with_error(domain: SimulationDomain) -> None:
    with pytest.raises(FileNotFoundError):
        InteractionModel(
            "tests/data/interaction_models/invalid_file.yaml",
            domain,
            ANNOTATED_CONVERSATIONS,
        )


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


def test_initialize_transition_matrices(im_crsv1: InteractionModel) -> None:
    (
        transition_single_intent,
        transition_compound_intent,
    ) = im_crsv1.initialize_transition_matrices(ANNOTATED_CONVERSATIONS)

    assert transition_single_intent.shape == (2, 3)
    assert transition_compound_intent.shape == (2, 4)

    assert transition_compound_intent.loc["ELICIT", "Intent-A_Intent-B"] == 0.5
    assert transition_single_intent.loc["INQUIRE", "Intent-A"] == 3 / 8


def test_get_next_dialogue_acts(im_crsv1: InteractionModel) -> None:
    dialogue_acts = im_crsv1.get_next_dialogue_acts(3)
    assert len(dialogue_acts) == 3
    assert dialogue_acts[0].intent == im_crsv1.INTENT_START
    assert dialogue_acts[1].intent == im_crsv1.INTENT_DISCLOSE
    assert dialogue_acts[2] == DialogueAct(
        im_crsv1.INTENT_DISCLOSE,
        [SlotValueAnnotation("DIRECTOR", "Steven Spielberg")],
    )


def test_intent_types(im_crsv1: InteractionModel) -> None:
    """Tests methods checking the type of intents."""
    assert im_crsv1.is_agent_intent_elicit(Intent("ELICIT")) is False
    assert im_crsv1.is_agent_intent_inquire(Intent("INQUIRE")) is True
    assert im_crsv1.is_agent_intent_set_retrieval(Intent("REVEAL")) is True


def test_is_transition_allowed(monkeypatch, im_crsv1: InteractionModel) -> None:
    monkeypatch.setattr(
        im_crsv1, "_current_dialogue_acts", [DialogueAct(Intent("DISCLOSE"))]
    )
    dialogue_acts_allowed = [
        DialogueAct(Intent("INQUIRE")),
        DialogueAct(Intent("REVEAL")),
    ]
    dialogue_acts_not_allowed = [DialogueAct(Intent("END"))]

    assert im_crsv1._is_transition_allowed(dialogue_acts_allowed) is True
    assert im_crsv1._is_transition_allowed(dialogue_acts_not_allowed) is False


def test_sample_next_user_dialogue_acts(
    caplog, im_crsv1: InteractionModel, information_need: InformationNeed
) -> None:
    agent_dialogue_acts = [
        DialogueAct(Intent("GREETING")),
        DialogueAct(Intent("ELICIT")),
    ]
    user_dialogue_acts = im_crsv1._sample_next_user_dialogue_acts(
        information_need, agent_dialogue_acts
    )
    assert len(user_dialogue_acts) == 1
    assert (
        "Transition matrix does not contain agent intent: GREETING"
        in caplog.text
    )

    agent_dialogue_acts = [
        DialogueAct(Intent("ELICIT")),
    ]
    user_dialogue_acts = im_crsv1._sample_next_user_dialogue_acts(
        information_need, agent_dialogue_acts
    )
    # The sampling is non-deterministic, so different outcomes are possible.
    # Here we check for the two possible outcomes.
    assert user_dialogue_acts == [
        DialogueAct(Intent("Intent-A")),
        DialogueAct(Intent("Intent-B")),
    ] or user_dialogue_acts == [DialogueAct(Intent("Intent-C"))]
