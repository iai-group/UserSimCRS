"""Tests for TUS's feature handler."""

from typing import List

import pytest
import torch
from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.annotation import Annotation
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.intent import Intent
from dialoguekit.participant import DialogueParticipant

from usersimcrs.core.information_need import InformationNeed
from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.dialogue_management.dialogue_state import DialogueState
from usersimcrs.simulator.neural.tus.tus_feature_handler import (
    TUSFeatureHandler,
)


@pytest.fixture
def feature_handler() -> TUSFeatureHandler:
    """Returns the feature handler."""
    _feature_handler = TUSFeatureHandler(
        domain=SimulationDomain("tests/data/domains/movies.yaml"),
        max_turn_feature_length=40,
        context_depth=2,
        user_actions=["inform", "request"],
        agent_actions=["elicit", "recommend", "bye"],
    )

    assert _feature_handler._user_actions == ["inform", "request"]
    assert _feature_handler._agent_actions == [
        "elicit",
        "recommend",
        "bye",
    ]

    return _feature_handler


def test__create_slot_index(feature_handler: TUSFeatureHandler) -> None:
    """Tests the creation of the slot index."""
    assert all(
        slot in feature_handler.slot_index.keys()
        for slot in [
            "TITLE",
            "GENRE",
            "ACTOR",
            "KEYWORD",
            "DIRECTOR",
            "PLOT",
            "RATING",
        ]
    )


@pytest.mark.parametrize(
    "slot, previous_state, state, expected_representation",
    [
        (
            "DIRECTOR",
            DialogueState(),
            DialogueState(),
            [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
        ),
        (
            "DIRECTOR",
            DialogueState(),
            DialogueState(
                user_dialogue_acts=[
                    DialogueAct(
                        Intent("inform"),
                        [Annotation("DIRECTOR", "Steven Spielberg")],
                    )
                ],
                belief_state={"DIRECTOR": "Steven Spielberg"},
            ),
            [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1],
        ),
        (
            "KEYWORD",
            DialogueState(),
            DialogueState(
                agent_dialogue_acts=[
                    DialogueAct(Intent("elicit"), [Annotation("KEYWORD")])
                ],
                belief_state={"KEYWORD": None},
            ),
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        ),
    ],
)
def test_get_basic_information_feature(
    feature_handler: TUSFeatureHandler,
    information_need: InformationNeed,
    slot: str,
    previous_state: DialogueState,
    state: DialogueState,
    expected_representation: List[int],
) -> None:
    """Tests the basic information feature."""
    assert (
        feature_handler.get_basic_information_feature(
            slot, information_need, state, previous_state
        )
        == expected_representation
    )


@pytest.mark.parametrize(
    "dialogue_acts, expected_representation",
    [
        ([], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
        (
            [DialogueAct(Intent("elicit"), [Annotation("GENRE")])],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
        ),
        (
            [
                DialogueAct(
                    Intent("recommend"), [Annotation("TITLE", "The Godfather")]
                ),
                DialogueAct(Intent("bye")),
            ],
            [0, 0, 0, 0, 0, 1, 1, 0, 0],
        ),
    ],
)
def test_get_agent_action_feature(
    dialogue_acts: List[DialogueAct],
    expected_representation: List[int],
    feature_handler: TUSFeatureHandler,
) -> None:
    """Tests the agent action feature."""
    assert (
        feature_handler.get_agent_action_feature(dialogue_acts)
        == expected_representation
    )


def test_get_slot_index_feature(feature_handler: TUSFeatureHandler) -> None:
    """Tests the slot index feature."""
    slots = list(feature_handler.slot_index.keys())
    i = slots.index("GENRE")
    expected = [0] * len(slots)
    expected[i] = 1
    assert feature_handler.get_slot_index_feature("GENRE") == expected


@pytest.mark.parametrize(
    "user_action_vector",
    [
        None,
        torch.tensor([0, 1, 0, 0, 0, 0]),
    ],
)
def test_get_slot_feature_vector(
    user_action_vector: torch.Tensor,
    feature_handler: TUSFeatureHandler,
    information_need: InformationNeed,
) -> None:
    """Tests the slot feature vector."""
    slot_feature_vector = feature_handler.get_slot_feature_vector(
        "DIRECTOR",
        DialogueState(),
        DialogueState(
            user_dialogue_acts=[
                DialogueAct(
                    Intent("inform"),
                    [Annotation("DIRECTOR", "Steven Spielberg")],
                )
            ],
            belief_state={"DIRECTOR": "Steven Spielberg"},
        ),
        information_need,
        [DialogueAct(Intent("elicit"), [Annotation("GENRE")])],
        user_action_vector,
    )
    user_action_vector = (
        user_action_vector
        if user_action_vector is not None
        else torch.tensor([0] * 6)
    )
    assert torch.equal(slot_feature_vector[23:29], user_action_vector)


@pytest.mark.parametrize(
    "utterance",
    [
        AnnotatedUtterance(
            "What genre are you interested in?",
            participant=DialogueParticipant.AGENT,
            intent=Intent("elicit"),
            annotations=[Annotation("GENRE")],
        ),
        AnnotatedUtterance(
            "Who should be the main actor?",
            participant=DialogueParticipant.AGENT,
            intent=Intent("elicit"),
            annotations=[Annotation("ACTOR")],
        ),
    ],
)
def test_build_input_vector(
    utterance: AnnotatedUtterance,
    feature_handler: TUSFeatureHandler,
    information_need: InformationNeed,
) -> None:
    """Tests the turn feature vector."""
    turn_vector, mask = feature_handler.build_input_vector(
        utterance,
        DialogueState(),
        DialogueState(),
        information_need,
        {"GENRE": torch.tensor([0, 1, 0, 0, 0, 0])},
    )
    assert len(turn_vector) == len(mask) == 40 * 2
    assert torch.equal(mask, torch.tensor([False] * 80))
