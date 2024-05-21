"""Tests for TUS's feature handler."""

from typing import Any, Dict, List

import pytest
from dialoguekit.core.annotation import Annotation
from dialoguekit.core.dialogue_act import DialogueAct

from usersimcrs.core.information_need import InformationNeed
from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.simulator.tus.feature_handler import FeatureHandler


@pytest.fixture
def information_need() -> InformationNeed:
    """Returns the information need."""
    return InformationNeed(
        {
            "DIRECTOR": "Steven Spielberg",
            "GENRE": "drama",
        },
        ["plot", "rating"],
    )


@pytest.fixture
def feature_handler() -> FeatureHandler:
    """Returns the feature handler."""
    _feature_handler = FeatureHandler(
        domain=SimulationDomain("tests/data/domains/movies.yaml"),
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


def test__create_slot_index(feature_handler: FeatureHandler) -> None:
    """Tests the creation of the slot index."""
    assert all(
        slot in feature_handler.slot_index.keys()
        for slot in [
            "TITLE",
            "GENRE",
            "ACTOR",
            "KEYWORD",
            "DIRECTOR",
            "plot",
            "rating",
        ]
    )


@pytest.mark.parametrize(
    "slot, previous_state, state, expected_representation",
    [
        ("DIRECTOR", {}, {}, [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0]),
        (
            "DIRECTOR",
            {},
            {"DIRECTOR": "Steven Spielberg"},
            [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1],
        ),
        ("plot", {}, {}, [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]),
    ],
)
def test_get_basic_information_feature(
    feature_handler: FeatureHandler,
    information_need: InformationNeed,
    slot: str,
    previous_state: Dict[str, Any],
    state: Dict[str, Any],
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
            [DialogueAct("elicit", [Annotation("GENRE")])],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
        ),
        (
            [
                DialogueAct(
                    "recommend", [Annotation("TITLE", "The Godfather")]
                ),
                DialogueAct("bye"),
            ],
            [0, 0, 0, 0, 0, 1, 1, 0, 0],
        ),
    ],
)
def test_get_agent_action_feature(
    dialogue_acts: List[DialogueAct],
    expected_representation: List[int],
    feature_handler: FeatureHandler,
) -> None:
    """Tests the agent action feature."""
    assert (
        feature_handler.get_agent_action_feature(dialogue_acts)
        == expected_representation
    )


def test_get_slot_index_feature(feature_handler: FeatureHandler) -> None:
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
        [0, 1, 0, 0, 0, 0],
    ],
)
def test_get_slot_feature_vector(
    user_action_vector: List[int],
    feature_handler: FeatureHandler,
    information_need: InformationNeed,
) -> None:
    """Tests the slot feature vector."""
    slot_feature_vector = feature_handler.get_slot_feature_vector(
        "DIRECTOR",
        {},
        {
            "DIRECTOR": "Steven Spielberg",
        },
        information_need,
        [DialogueAct("elicit", [Annotation("GENRE")])],
        user_action_vector,
    )
    user_action_vector = user_action_vector if user_action_vector else [0] * 6
    assert slot_feature_vector[21:27] == user_action_vector


def test_get_turn_feature_vector(
    feature_handler: FeatureHandler,
    information_need: InformationNeed,
) -> None:
    """Tests the turn feature vector."""
    turn_vector = feature_handler.get_turn_feature_vector(
        ["DIRECTOR", "GENRE"],
        {},
        {},
        information_need,
        [DialogueAct("elicit", [Annotation("GENRE")])],
        [None, [0, 1, 0, 0, 0, 0]],
    )
    assert len(turn_vector) == 2
    assert len(turn_vector[0]) == len(turn_vector[1]) == 34
