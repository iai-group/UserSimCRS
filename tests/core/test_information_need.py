"""Tests for the InformationNeed class."""

import pytest

from usersimcrs.core.information_need import (
    InformationNeed,
    generate_random_information_need,
)
from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item import Item
from usersimcrs.items.item_collection import ItemCollection


def test_generate_random_information_need(
    domain: SimulationDomain, item_collection: ItemCollection
) -> None:
    """Test generate_information_need.

    Args:
        domain: Simulation domain.
        item_collection: Item collection.
    """
    information_need = generate_random_information_need(domain, item_collection)
    assert all(information_need.constraints.values())
    assert all(
        [
            slot in domain.get_requestable_slots()
            for slot in information_need.requested_slots
        ]
    )
    assert len(information_need.target_items) == 1


@pytest.mark.parametrize(
    "slot,expected_value",
    [
        ("GENRE", "Comedy"),
        ("DIRECTOR", "Steven Spielberg"),
        ("KEYWORDS", None),
    ],
)
def test_get_constraint_value(
    information_need: InformationNeed, slot: str, expected_value: str
) -> None:
    """Test get_constraint_value.

    Args:
        information_need: Information need.
        slot: Slot.
        expected_value: Expected value.
    """
    assert information_need.get_constraint_value(slot) == expected_value


def test_get_requestable_slots(information_need: InformationNeed) -> None:
    """Test get_requestable_slots.

    Args:
        information_need: Information need.
    """
    assert information_need.get_requestable_slots() == ["PLOT", "RATING"]
    information_need.requested_slots["RATING"] = 4.5
    assert information_need.get_requestable_slots() == ["PLOT"]


def test_to_dict(information_need: InformationNeed) -> None:
    """Test to_dict.

    Args:
        information_need: Information need.
    """
    assert information_need.to_dict() == {
        "target_items": [
            {
                "item_id": "1",
                "properties": {
                    "GENRE": "Comedy",
                    "DIRECTOR": "Steven Spielberg",
                    "RATING": 4.5,
                    "PLOT": "A movie plot",
                },
            }
        ],
        "constraints": {"GENRE": "Comedy", "DIRECTOR": "Steven Spielberg"},
        "requests": ["PLOT", "RATING"],
    }


def test_from_dict() -> None:
    """Tests from_dict."""
    data = {
        "target_items": [
            {
                "item_id": "1",
                "properties": {
                    "GENRE": "Comedy",
                    "DIRECTOR": "Steven Spielberg",
                    "PLOT": "A movie plot",
                },
            },
            {
                "item_id": "2",
                "properties": {
                    "GENRE": "Drama",
                    "ACTOR": "Steven Spielberg",
                    "RATING": 4.5,
                },
            },
        ],
        "constraints": {"GENRE": "Comedy", "DIRECTOR": "Steven Spielberg"},
        "requests": ["PLOT", "RATING"],
    }
    expected_information_need = InformationNeed(
        [
            Item(
                "1",
                {
                    "GENRE": "Comedy",
                    "DIRECTOR": "Steven Spielberg",
                    "PLOT": "A movie plot",
                },
            ),
            Item(
                "2",
                {"GENRE": "Drama", "ACTOR": "Steven Spielberg", "RATING": 4.5},
            ),
        ],
        {"GENRE": "Comedy", "DIRECTOR": "Steven Spielberg"},
        ["PLOT", "RATING"],
    )
    loaded_information_need = InformationNeed.from_dict(data)
    assert [target.id for target in loaded_information_need.target_items] == [
        target.id for target in expected_information_need.target_items
    ]
    assert (
        loaded_information_need.constraints
        == expected_information_need.constraints
    )
    assert (
        loaded_information_need.requested_slots
        == expected_information_need.requested_slots
    )
