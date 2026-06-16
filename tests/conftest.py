"""Fixtures for the tests."""

import os
from typing import List
from unittest.mock import MagicMock

import pytest

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.intent import Intent
from dialoguekit.utils.dialogue_reader import json_to_dialogues

from usersimcrs.core.information_need import InformationNeed
from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item import Item
from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.llm_interfaces.ollama_interface import OllamaLLMInterface

DOMAIN_YAML_FILE = "tests/data/domains/movies.yaml"
ITEMS_CSV_FILE = "tests/data/items/movies_w_keywords.csv"


@pytest.fixture(scope="session")
def domain() -> SimulationDomain:
    """Domain fixture."""
    return SimulationDomain(DOMAIN_YAML_FILE)


@pytest.fixture(scope="module")
def information_need() -> InformationNeed:
    """Information need fixture."""
    constraints = {"GENRE": "Comedy", "DIRECTOR": "Steven Spielberg"}
    requests = ["PLOT", "RATING"]
    target_items = [
        Item(
            "1",
            {
                "GENRE": "Comedy",
                "DIRECTOR": "Steven Spielberg",
                "RATING": 4.5,
                "PLOT": "A movie plot",
            },
        )
    ]
    return InformationNeed(target_items, constraints, requests)


@pytest.fixture(scope="session")
def item_collection(domain: SimulationDomain):
    """Item collection fixture."""
    item_collection = ItemCollection("tests/data/items.db", "test_collection")
    mapping = {
        "title": {"slot": "TITLE"},
        "genres": {
            "slot": "GENRE",
            "multi-valued": True,
            "delimiter": "|",
        },
        "keywords": {
            "slot": "KEYWORD",
            "multi-valued": True,
            "delimiter": "|",
        },
    }
    item_collection.load_items_csv(
        ITEMS_CSV_FILE,
        id_col="movieId",
        domain=domain,
        domain_mapping=mapping,
    )
    yield item_collection
    os.remove("tests/data/items.db")


@pytest.fixture
def dialogues() -> List[Dialogue]:
    """Loads annotated test dialogues."""
    return json_to_dialogues(
        "tests/data/annotated_dialogues.json",
        agent_ids=["Agent"],
        user_ids=["User"],
    )


@pytest.fixture
def mock_ollama_interface() -> OllamaLLMInterface:
    """Mock Ollama LLM interface fixture."""
    return MagicMock(spec=OllamaLLMInterface)


@pytest.fixture
def recommendation_intents() -> List[Intent]:
    """Recommendation intent fixture."""
    return [Intent("REC-S"), Intent("REC-E")]


@pytest.fixture
def acceptance_intents() -> List[Intent]:
    """Acceptance intent fixture."""
    return [Intent("ACC")]


@pytest.fixture
def rejection_intents() -> List[Intent]:
    """Rejection intent fixture."""
    return [Intent("REJ")]
