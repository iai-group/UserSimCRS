"""Tests for Ratings."""

import pytest
from dialoguekit.core.domain import Domain

from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.items.ratings import Ratings

DOMAIN_YAML_FILE = "tests/data/domains/movies.yaml"
DOMAIN_MAPPING = {
    "title": {"slot": "TITLE"},
    "genres": {
        "slot": "GENRE",
        "multi-valued": True,
        "delimiter": "|",
    },
}
ITEMS_CSV_FILE = "tests/data/items/movies.csv"
RATINGS_CSV_FILE = "tests/data/items/ratings.csv"


@pytest.fixture
def ratings() -> Ratings:
    """Ratings fixture."""
    domain = Domain(DOMAIN_YAML_FILE)
    item_collection = ItemCollection()
    item_collection.load_items_csv(
        ITEMS_CSV_FILE,
        id_col="movieId",
        domain=domain,
        domain_mapping=DOMAIN_MAPPING,
    )
    ratings = Ratings()
    ratings.load_ratings_csv(RATINGS_CSV_FILE)
    return ratings


@pytest.mark.parametrize(
    "historical_ratio",
    [0.5, 0.8, 0.2],
)
def test_create_split(historical_ratio: float, ratings: Ratings) -> None:
    """Tests create_split.

    Args:
        historical_ratio: Ratio of historical items.
    """
    nb_items = len(ratings._item_ratings)
    historical_ratings, ground_truth_ratings = ratings.create_split(
        historical_ratio
    )

    assert len(historical_ratings._item_ratings) == int(
        nb_items * historical_ratio
    )
    assert (
        len(ground_truth_ratings._item_ratings)
        == int(nb_items - nb_items * historical_ratio) + 1
    )


def test_create_split_error(ratings: Ratings) -> None:
    with pytest.raises(ValueError):
        ratings.create_split(2)
