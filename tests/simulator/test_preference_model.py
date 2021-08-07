"""Tests for PreferenceModel."""

import pytest

from cryses.simulator.preference_model import (
    PreferenceModel,
    PreferenceModelVariant,
)
from dialoguekit.core.ontology import Ontology
from dialoguekit.core.recsys.item_collection import ItemCollection
from dialoguekit.core.recsys.ratings import Ratings

ONTOLOGY_YAML_FILE = "../dialoguekit/tests/data/ontology.yaml"
ITEMS_CSV_FILE = "../dialoguekit/tests/data/movielens-20m-sample/movies.csv"
RATINGS_CSV_FILE = "../dialoguekit/tests/data/movielens-20m-sample/ratings.csv"


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


def test_initial_item_preferences_sip(preference_model_sip):
    # TODO this is the value in the original rating file, but it'll be a bit
    # tricky to test when only a sample of the original ratings is used.
    assert preference_model_sip.get_item_preference("356") == 1.0


# TODO Write tests for get_item_preference(), get_slotvalue_preference,
# and get_slot_preference()
# See: https://github.com/iai-group/cryses/issues/21
