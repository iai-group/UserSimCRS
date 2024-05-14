"""Tests for SimplePreferenceModel."""

import os

import pytest
from dialoguekit.core.domain import Domain

from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.items.ratings import Ratings
from usersimcrs.user_modeling.simple_preference_model import (
    SimplePreferenceModel,
)

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
def preference_model() -> SimplePreferenceModel:
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
    return SimplePreferenceModel(
        domain,
        item_collection,
        ratings,
        historical_user_id="13",
    )


@pytest.mark.parametrize("item_id, expected", [("377", True), ("591", False)])
def test_is_item_consumed(
    item_id: str, expected: bool, preference_model: SimplePreferenceModel
) -> None:
    assert expected == preference_model.is_item_consumed(item_id)


def test_get_item_preference(preference_model: SimplePreferenceModel) -> None:
    preference = preference_model.get_item_preference("527")
    assert preference == 1.0 or preference == -1.0


def test_get_item_preference_nonexisting_item(
    preference_model: SimplePreferenceModel,
) -> None:
    with pytest.raises(ValueError):
        preference_model.get_item_preference("1020")


def test_get_slot_value_preference(
    preference_model: SimplePreferenceModel,
) -> None:
    preference = preference_model.get_slot_value_preference(
        "GENRE", "Animation"
    )
    assert preference == 1.0 or preference == -1.0


def test_get_slot_value_preference_nonexisting_slot(
    preference_model: SimplePreferenceModel,
) -> None:
    with pytest.raises(ValueError):
        preference_model.get_slot_value_preference("WRITER", "Billy Wilder")


def test_get_slot_preference(preference_model: SimplePreferenceModel) -> None:
    value, preference = preference_model.get_slot_preference("GENRE")
    assert (
        value
        in preference_model._item_collection.get_possible_property_values(
            "GENRE"
        )
    )
    assert preference == 1


def test_get_slot_preference_nonexisting_slot(
    preference_model: SimplePreferenceModel,
) -> None:
    with pytest.raises(ValueError):
        preference_model.get_slot_preference("YEAR")


def test_load_preferences_error(
    preference_model: SimplePreferenceModel,
) -> None:
    with pytest.raises(FileNotFoundError):
        preference_model.load_preference_model("path")


def test_save_preference_model(
    preference_model: SimplePreferenceModel,
) -> None:
    preference_model._slot_value_preferences.set_preference(
        "GENRE", "Action", 1
    )
    preference_model._item_preferences.set_preference("ID", "527", -1)
    preference_model.save_preference_model("tests/data/preference_model.joblib")

    loaded_model = SimplePreferenceModel.load_preference_model(
        "tests/data/preference_model.joblib"
    )

    assert loaded_model._user_id == preference_model._user_id
    assert (
        loaded_model._item_preferences._preferences
        == preference_model._item_preferences._preferences
    )
    assert (
        loaded_model._slot_value_preferences._preferences
        == preference_model._slot_value_preferences._preferences
    )

    os.remove("tests/data/preference_model.joblib")
