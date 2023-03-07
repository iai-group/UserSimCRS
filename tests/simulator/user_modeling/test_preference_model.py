"""Tests for PreferenceModel."""

import pytest
from dialoguekit.core.domain import Domain

from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.items.ratings import Ratings
from usersimcrs.user_modeling.preference_model import PreferenceModel

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
def preference_model() -> PreferenceModel:
    """Preference model fixture."""
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
    return PreferenceModel(
        domain,
        item_collection,
        ratings,
        historical_user_id="13",
    )


@pytest.mark.parametrize("item_id, expected", [("377", True), ("591", False)])
def test_is_item_consumed(
    item_id: str, expected: bool, preference_model: PreferenceModel
) -> None:
    assert expected == preference_model.is_item_consumed(item_id)


def test_get_item_preference(preference_model: PreferenceModel) -> None:
    preference = preference_model.get_item_preference("527")
    assert preference == 1.0 or preference == -1.0


def test_get_item_preference_error(preference_model: PreferenceModel) -> None:
    with pytest.raises(ValueError):
        preference_model.get_item_preference("1020")


def test_get_slot_value_preference(preference_model: PreferenceModel) -> None:
    preference = preference_model.get_slot_value_preference(
        "GENRE", "Animation"
    )
    assert preference == 1.0 or preference == -1.0


def test_get_slot_value_preference_error(
    preference_model: PreferenceModel,
) -> None:
    with pytest.raises(ValueError):
        preference_model.get_slot_value_preference("WRITER", "Billy Wilder")


def test_get_slot_preference(preference_model: PreferenceModel) -> None:
    value, preference = preference_model.get_slot_preference("GENRE")
    assert (
        value
        in preference_model._item_collection.get_possible_property_values(
            "GENRE"
        )
    )
    assert preference == 1


def test_get_slot_preference_error(preference_model: PreferenceModel) -> None:
    with pytest.raises(ValueError):
        preference_model.get_slot_preference("YEAR")
