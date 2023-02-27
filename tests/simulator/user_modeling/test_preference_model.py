"""Tests for PreferenceModel."""

import pytest
from dialoguekit.core.domain import Domain

from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.items.ratings import Ratings
from usersimcrs.user_modeling.preference_model import (
    PreferenceModel,
    PreferenceModelVariant,
)

DOMAIN_YAML_FILE = "data/domains/movies.yaml"
DOMAIN_MAPPING = {
    "movieId": {"slot": "ID"},
    "title": {"slot": "TITLE"},
    "genres": {
        "slot": "GENRE",
        "multi-valued": True,
        "delimiter": "|",
    },
}
ITEMS_CSV_FILE = "data/movielens-20m-sample/movies.csv"
RATINGS_CSV_FILE = "data/movielens-20m-sample/ratings.csv"


# Single item preference model variant.
@pytest.fixture
def preference_model_sip() -> PreferenceModel:
    domain = Domain(DOMAIN_YAML_FILE)
    item_collection = ItemCollection()
    item_collection.load_items_csv(
        ITEMS_CSV_FILE, domain=domain, domain_mapping=DOMAIN_MAPPING
    )
    ratings = Ratings()
    ratings.load_ratings_csv(RATINGS_CSV_FILE)
    return PreferenceModel(
        domain,
        item_collection,
        ratings,
        PreferenceModelVariant.SIP,
        historical_user_id="13",
    )


# Peronal Knowledge Graph preference model variant.
@pytest.fixture
def preference_model_pkg() -> PreferenceModel:
    domain = Domain(DOMAIN_YAML_FILE)
    item_collection = ItemCollection()
    item_collection.load_items_csv(
        ITEMS_CSV_FILE, domain=domain, domain_mapping=DOMAIN_MAPPING
    )
    ratings = Ratings()
    ratings.load_ratings_csv(RATINGS_CSV_FILE)
    return PreferenceModel(
        domain,
        item_collection,
        ratings,
        PreferenceModelVariant.PKG,
        historical_user_id="13",
    )


def test_initial_item_preferences_sip(preference_model_sip) -> None:
    # TODO this is the value in the original rating file, but it'll be a bit
    # tricky to test when only a sample of the original ratings is used.
    assert preference_model_sip.get_item_preference("356") == 1.0


def test_get_slotvalue_preference_sip(preference_model_sip) -> None:
    preference = preference_model_sip.get_slotvalue_preference(
        "GENRE", "Comedy"
    )
    assert -1 <= preference <= 1


def test_get_slotvalue_preference_pkg(preference_model_pkg) -> None:
    preference = preference_model_pkg.get_slotvalue_preference("GENRE", "Drama")
    assert preference == 0.6


def test_get_item_preference_sip(preference_model_sip) -> None:
    preference = preference_model_sip.get_item_preference("10")
    assert -1 <= preference <= 1


def test_get_item_preference_pkg(preference_model_pkg):
    # When (Both Drama and Thriller are favored as 1)
    preference = preference_model_pkg.get_item_preference("100")
    assert preference == 1.0


def test_get_slot_preference_pkg(preference_model_pkg):
    slot, preference = preference_model_pkg.get_slot_preference("GENRE")
    assert isinstance(slot, str)
    assert -1 <= preference <= 1


# TODO Write tests for get_item_preference(), get_slotvalue_preference,
# and get_slot_preference()
# See: https://github.com/iai-group/UserSimCRS/issues/21
