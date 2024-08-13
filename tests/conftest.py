"""Fixtures for the tests."""

import os

import pytest

from usersimcrs.core.information_need import InformationNeed
from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item import Item
from usersimcrs.items.item_collection import ItemCollection

DOMAIN_YAML_FILE = "tests/data/domains/movies.yaml"
ITEMS_CSV_FILE = "tests/data/items/movies_w_keywords.csv"


@pytest.fixture(scope="session")
def domain() -> SimulationDomain:
    """Domain fixture."""
    return SimulationDomain(DOMAIN_YAML_FILE)


@pytest.fixture(scope="session")
def item_collection(domain: SimulationDomain):
    """Item collection fixture."""
    item_collection = ItemCollection("tests/data/items.db", "test_collection")
    mapping = {
        "title": {"slot": "TITLE"},
        "genres": {
            "slot": "GENRE",
            "multi-valued": True,
            "delimiter": "|",
        },
        "keywords": {
            "slot": "KEYWORD",
            "multi-valued": True,
            "delimiter": "|",
        },
    }
    item_collection.load_items_csv(
        ITEMS_CSV_FILE,
        id_col="movieId",
        domain=domain,
        domain_mapping=mapping,
    )
    yield item_collection
    os.remove("tests/data/items.db")


@pytest.fixture
def information_need() -> InformationNeed:
    """Information need fixture."""
    constraints = {"GENRE": "Comedy", "DIRECTOR": "Steven Spielberg"}
    requests = ["PLOT", "RATING"]
    target_items = [
        Item(
            "1",
            {
                "GENRE": "Comedy",
                "DIRECTOR": "Steven Spielberg",
                "RATING": 4.5,
                "PLOT": "A movie plot",
            },
        )
    ]
    return InformationNeed(target_items, constraints, requests)
