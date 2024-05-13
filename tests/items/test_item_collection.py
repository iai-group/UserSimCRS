"""Tests for ItemCollection."""

from typing import Any, Dict, List

import pytest
from dialoguekit.core.annotation import Annotation
from dialoguekit.core.domain import Domain

from usersimcrs.items.item_collection import ItemCollection


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
        "YEAR": "1995",
    }


def test_load_items_csv(
    item_collection: ItemCollection, movie: Dict[str, Any]
) -> None:
    """Tests items loading with a domain and mapping."""

    item = item_collection.get_item(movie["ID"])

    for property in ["TITLE", "GENRE"]:
        assert item.get_property(property) == movie[property]
    assert item.get_property("YEAR") is None


def test_get_possible_property_values(
    item_collection: ItemCollection,
) -> None:
    """Tests using slot with different types (str, list) and unknown slot."""

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


@pytest.mark.parametrize(
    "annotations, expected_num_matching_items",
    [
        ([], 0),
        ([Annotation("GENRE", "Adventure")], 1507),
        (
            [Annotation("GENRE", "Adventure"), Annotation("GENRE", "Comedy")],
            464,
        ),
    ],
)
def test_get_items_by_properties(
    item_collection: ItemCollection,
    annotations: List[Annotation],
    expected_num_matching_items: int,
) -> None:
    """Tests getting items by properties."""
    matching_items = item_collection.get_items_by_properties(annotations)
    assert len(matching_items) == expected_num_matching_items
