"""Tests for Ratings."""

from typing import Dict, List

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


def simple_user_item_sampler(
    item_ratings: Dict[str, float],
    historical_ratio: float,
) -> List[str]:
    """Samples the first portion of items rated by a given user.

    Args:
        item_ratings: Item ratings to sample.
        historical_ratio: Ratio of items ratings to be used as historical
          data.

    Returns:
        List of sampled item ids.
    """
    # Determine the number of items to use as historical data for a given user.
    nb_historical_items = int(historical_ratio * len(item_ratings))
    return list(item_ratings.keys())[:nb_historical_items]


@pytest.fixture
def ratings() -> Ratings:
    """Ratings fixture."""
    domain = Domain(DOMAIN_YAML_FILE)
    item_collection = ItemCollection("tests/data/items.db", "test_ratings")
    item_collection.load_items_csv(
        ITEMS_CSV_FILE,
        id_col="movieId",
        domain=domain,
        domain_mapping=DOMAIN_MAPPING,
    )
    ratings = Ratings(item_collection)
    ratings.load_ratings_csv(RATINGS_CSV_FILE)
    return ratings


@pytest.mark.parametrize(
    "historical_ratio",
    [0.5, 0.8, 0.2],
)
def test_user_item_sampler_ratio(
    historical_ratio: float, ratings: Ratings
) -> None:
    user_id = "1"
    user_ratings = ratings.get_user_ratings(user_id)

    item_sample = user_item_sampler(
        user_ratings,
        historical_ratio,
    )

    assert len(item_sample) == int(len(user_ratings) * historical_ratio)


@pytest.mark.parametrize(
    "historical_ratio, user_id,historical_item_id, ground_truth_item_id",
    [
        (0.5, "1", "29", "367"),
        (0.8, "1", "32", "919"),
        (0.8, "23", "293", "838"),
        (0.2, "5", "60", "377"),
    ],
)
def test_create_split(
    historical_ratio: float,
    user_id: str,
    historical_item_id: str,
    ground_truth_item_id: str,
    ratings: Ratings,
) -> None:
    """Tests create_split method with a simple sampler.

    Args:
        historical_ratio: Ratio of historical items.
        user_id: User id.
        historical_item_id: Id of a historical item for user_id.
        ground_truth_item_id: Id of an unseen item ofr user_id.
    """
    historical_ratings, ground_truth_ratings = ratings.create_split(
        historical_ratio, simple_user_item_sampler
    )

    assert set(historical_ratings.get_user_ratings(user_id).keys()).isdisjoint(
        set(ground_truth_ratings.get_user_ratings(user_id))
    )

    assert historical_ratings.get_user_item_rating(user_id, historical_item_id)
    assert ground_truth_ratings.get_user_item_rating(
        user_id, ground_truth_item_id
    )


def test_create_split_ratio_error(ratings: Ratings) -> None:
    with pytest.raises(ValueError):
        ratings.create_split(2)


def test_add_user_item_rating_nonexistent_item(ratings: Ratings) -> None:
    user_id = "1"
    item_id = "1342"
    rating = 0.3

    original_item_ratings = ratings.get_item_ratings(item_id)
    original_user_ratings = ratings.get_user_ratings(user_id)

    ratings.add_user_item_rating(user_id, item_id, rating)

    assert original_item_ratings == ratings.get_item_ratings(item_id)
    assert original_user_ratings == ratings.get_user_ratings(user_id)
