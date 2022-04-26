"""Tests for PreferenceModel."""

import pytest

from usersimcrs.simulator.preference_model import (
    PreferenceModel,
    PreferenceModelVariant,
)
from dialoguekit.core.ontology import Ontology
from dialoguekit.core.recsys.item_collection import ItemCollection
from dialoguekit.core.recsys.ratings import Ratings

ONTOLOGY_YAML_FILE = "data/ontology.yaml"
ITEMS_CSV_FILE = "data/movielens-20m-sample/movies.csv"
RATINGS_CSV_FILE = "data/movielens-20m-sample/ratings.csv"


# Single item preference model variant.
@pytest.fixture
def preference_model_sip():
    ontology = Ontology(ONTOLOGY_YAML_FILE)
    item_collection = ItemCollection()
    item_collection.load_items_csv(ITEMS_CSV_FILE, ["ID", "NAME", "genres"])
    ratings = Ratings()
    ratings.load_ratings_csv(RATINGS_CSV_FILE)
    return PreferenceModel(
        ontology,
        item_collection,
        ratings,
        PreferenceModelVariant.SIP,
        historical_user_id="13",
    )


# Peronal Knowledge Graph preference model variant.
@pytest.fixture
def preference_model_pkg():
    ontology = Ontology(ONTOLOGY_YAML_FILE)
    item_collection = ItemCollection()
    item_collection.load_items_csv(ITEMS_CSV_FILE, ["ID", "NAME", "genres"])
    ratings = Ratings()
    ratings.load_ratings_csv(RATINGS_CSV_FILE)
    return PreferenceModel(
        ontology,
        item_collection,
        ratings,
        PreferenceModelVariant.PKG,
        historical_user_id="13",
    )


def test_initial_item_preferences_sip(preference_model_sip):
    # TODO this is the value in the original rating file, but it'll be a bit
    # tricky to test when only a sample of the original ratings is used.
    assert preference_model_sip.get_item_preference("356") == 1.0


def test_get_slotvalue_preference_sip(preference_model_sip):
    # Given
    slot = "GENRE"
    value = "Comedy"

    # When
    preference = preference_model_sip.get_slotvalue_preference(slot, value)

    # Then
    assert -1 <= preference <= 1


def test_get_slotvalue_preference_pkg(preference_model_pkg):
    # Given
    slot = "GENRE"
    value = "Drama"

    # When
    preference = preference_model_pkg.get_slotvalue_preference(slot, value)

    # Then
    assert preference == 1


def test_get_item_preference_sip(preference_model_sip):
    # Given
    item_id = "10"

    # When
    preference = preference_model_sip.get_item_preference(item_id)

    # Then
    assert -1 <= preference <= 1


def test_get_item_preference_pkg(preference_model_pkg):
    # Given
    item_id = "100"  # geners = ['Drama', 'Thriller']

    # When (Both Drama and Thriller are favored as 1)
    preference = preference_model_pkg.get_item_preference(item_id)

    # Then
    assert preference == 1


def test_get_slot_preference_pkg(preference_model_pkg):
    # Given
    slot = "GENRE"

    # When
    slot, preference = preference_model_pkg.get_slot_preference(slot)

    # Then
    assert isinstance(slot, str)
    assert preference in [-1, 0, 1]


# TODO Write tests for get_item_preference(), get_slotvalue_preference,
# and get_slot_preference()
# See: https://github.com/iai-group/UserSimCRS/issues/21
