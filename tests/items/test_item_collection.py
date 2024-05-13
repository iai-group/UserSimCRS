"""Tests for ItemCollection."""

from typing import Any, Dict

import pytest

from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item_collection import ItemCollection

DOMAIN_YAML_FILE = "tests/data/domains/movies.yaml"
ITEMS_CSV_FILE = "tests/data/items/movies_w_keywords.csv"


@pytest.fixture
def domain() -> SimulationDomain:
    """Domain fixture."""
    return SimulationDomain(DOMAIN_YAML_FILE)


@pytest.fixture
def movie() -> Dict[str, Any]:
    """Movie fixture representing the first item of the collection."""
    return {
        "ID": "1",
        "TITLE": "Toy Story (1995)",
        "GENRE": ["Adventure", "Animation", "Children", "Comedy", "Fantasy"],
        "KEYWORD": [
            "animation",
            "kids and family",
            "pixar animation",
            "computer animation",
            "toys",
        ],
    }


def test_load_items_csv(
    domain: SimulationDomain, movie: Dict[str, Any]
) -> None:
    """Tests items loading with a domain and mapping."""
    item_collection = ItemCollection()
    mapping = {
        "title": {"slot": "TITLE"},
        "genres": {
            "slot": "GENRE",
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

    item = item_collection.get_item(movie["ID"])

    for property in ["TITLE", "GENRE"]:
        assert item.get_property(property) == movie[property]
    assert item.get_property("KEYWORD") is None


def test_get_possible_property_values(domain: SimulationDomain) -> None:
    """Tests using slot with different types (str, list) and unknown slot."""
    item_collection = ItemCollection()
    mapping = {
        "title": {"slot": "TITLE"},
        "genres": {
            "slot": "GENRE",
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

    genres = item_collection.get_possible_property_values("GENRE")
    assert len(genres) == 20
    assert {
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Fantasy",
    }.issubset(genres)
    assert not {"Biography", "Short Film"}.issubset(genres)

    titles = item_collection.get_possible_property_values("TITLE")
    assert len(titles) == 13813
    assert "Toy Story (1995)" in titles
    assert "Toy Story 4" not in titles

    unknown_property = item_collection.get_possible_property_values("UNKNOWN")
    assert len(unknown_property) == 0
