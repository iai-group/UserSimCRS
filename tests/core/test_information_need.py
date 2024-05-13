"""Tests for the InformationNeed class."""

import pytest

from usersimcrs.core.information_need import (
    InformationNeed,
    generate_information_need,
)
from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item_collection import ItemCollection

DOMAIN_YAML_FILE = "tests/data/domains/movies.yaml"
ITEMS_CSV_FILE = "tests/data/items/movies_w_keywords.csv"


@pytest.fixture
def domain() -> SimulationDomain:
    """Domain fixture."""
    return SimulationDomain(DOMAIN_YAML_FILE)


@pytest.fixture
def item_collection(domain: SimulationDomain) -> ItemCollection:
    """Item collection fixture."""
    mapping = {
        "title": {"slot": "TITLE"},
        "genres": {
            "slot": "GENRE",
            "multi-valued": True,
            "delimiter": "|",
        },
    }

    item_collection = ItemCollection()
    item_collection.load_items_csv(
        ITEMS_CSV_FILE,
        id_col="movieId",
        domain=domain,
        domain_mapping=mapping,
    )
    return item_collection


def test_generate_information_need(
    domain: SimulationDomain, item_collection: ItemCollection
) -> None:
    """Test generate_information_need.

    Args:
        domain: Simulation domain.
        item_collection: Item collection.
    """
    information_need = generate_information_need(domain, item_collection)
    assert all(information_need.constraints.values())
    assert all(
        [
            slot in domain.get_requestable_slots()
            for slot in information_need.requested_slots
        ]
    )


@pytest.fixture
def information_need() -> InformationNeed:
    """Information need fixture."""
    constraints = {"GENRE": "Comedy", "DIRECTOR": "Steven Spielberg"}
    requests = ["plot", "rating"]
    return InformationNeed(constraints, requests)


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
    assert information_need.get_requestable_slots() == ["plot", "rating"]
    information_need.requested_slots["rating"] = 4.5
    assert information_need.get_requestable_slots() == ["plot"]
