"""Fixtures for the tests."""

import os

import pytest
from dialoguekit.core.domain import Domain

from usersimcrs.items.item_collection import ItemCollection

DOMAIN_YAML_FILE = "tests/data/domains/movies.yaml"
ITEMS_CSV_FILE = "tests/data/items/movies_w_keywords.csv"


@pytest.fixture(scope="session")
def domain() -> Domain:
    """Domain fixture."""
    return Domain(DOMAIN_YAML_FILE)


@pytest.fixture(scope="session")
def item_collection(domain: Domain):
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
    item_collection.close()
    os.remove("tests/data/items.db")
