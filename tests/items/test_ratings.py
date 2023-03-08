"""Tests for Ratings."""

import pytest
from dialoguekit.core.domain import Domain

from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.items.ratings import Ratings, user_item_sampler

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
def test_create_split_user_item_sampler(
    historical_ratio: float, ratings: Ratings
) -> None:
    """Tests create_split with `user_item_sampler` method.

    Args:
        historical_ratio: Ratio of historical items.
    """
    historical_ratings, ground_truth_ratings = ratings.create_split(
        historical_ratio, user_item_sampler
    )

    for user_id, user_ratings in ratings._user_ratings.items():
        historical_user_ratings = historical_ratings.get_user_ratings(user_id)
        ground_truth_user_ratings = ground_truth_ratings.get_user_ratings(
            user_id
        )
        assert len(historical_user_ratings) == int(
            len(user_ratings) * historical_ratio
        )
        assert len(ground_truth_user_ratings) == len(user_ratings) - len(
            historical_ratings.get_user_ratings(user_id)
        )
        assert set(
            historical_ratings.get_user_ratings(user_id).keys()
        ).isdisjoint(set(ground_truth_ratings.get_user_ratings(user_id)))


def test_create_split_ratio_error(ratings: Ratings) -> None:
    with pytest.raises(ValueError):
        ratings.create_split(2)
