"""Tests for SimplePreferenceModel."""

import os

import pytest

from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.items.ratings import Ratings
from usersimcrs.user_modeling.simple_preference_model import (
    SimplePreferenceModel,
)

RATINGS_CSV_FILE = "tests/data/items/ratings.csv"


@pytest.fixture
def preference_model(
    domain: SimulationDomain, item_collection: ItemCollection
) -> SimplePreferenceModel:
    """Preference model fixture."""
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
        preference_model.get_item_preference("23043")


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
