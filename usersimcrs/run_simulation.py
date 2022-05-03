"""Console application for running simulation."""

import argparse
import os
import sys
import json

from dialoguekit.nlu.models.diet_classifier_rasa import IntentClassifierRasa
from dialoguekit.nlg.nlg import NLG
from dialoguekit.agent.agent import Agent
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.ontology import Ontology
from dialoguekit.core.recsys.item_collection import ItemCollection
from dialoguekit.core.recsys.ratings import Ratings
from dialoguekit.manager.dialogue_manager import DialogueManager

from usersimcrs.simulator.user_simulator import UserSimulator
from usersimcrs.sample_agent.sample_agent import SampleAgent
from usersimcrs.simulator.agenda_based_simulator import AgendaBasedSimulator
from usersimcrs.simulator.preference_model import (
    PreferenceModel,
    PreferenceModelVariant,
)
from usersimcrs.simulator.interaction_model import InteractionModel


def simulate_conversation(
    agent: Agent, user_simulator: UserSimulator
) -> Dialogue:
    """Simulates a single conversation and returns the resulting dialogue.

    Args:
        agent: An agent.
        user_simulator: A user simulator.
    """
    platform = None  # TODO: Add simulator platform
    dm = DialogueManager(agent, user_simulator, platform)
    dm.start()
    dm.close()
    return dm.dialogue_history


if __name__ == "__main__":
    agent = SampleAgent(agent_id="sample_agent")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ontology", type=str, help="Path to ontology config file."
    )
    parser.add_argument("-items", type=str, help="Path to items file.")
    parser.add_argument("-ratings", type=str, help="Path to ratings file.")
    args = parser.parse_args()

    # TODO Load settings from command line arguments or config file.
    if not os.path.exists(args.ontology):
        sys.exit("FileNotFound: ", args.ontology)
    if not os.path.exists(args.items):
        sys.exit("FileNotFound: ", args.items)
    if not os.path.exists(args.ratings):
        sys.exit("FileNotFound: ", args.ratings)

    ontology = Ontology(args.ontology)

    item_collection = ItemCollection()
    item_collection.load_items_csv(args.items, ["ID", "NAME", "genres"])

    ratings = Ratings(item_collection)
    ratings.load_ratings_csv(args.ratings)

    annotated_dialogues_file = open(
        "data/agents/moviebot/annotated_dialogues.json"
    )
    annotated_conversations = json.load(annotated_dialogues_file)

    # TODO: initialization of the simulator with NLU, NLG, etc.
    preference_model = PreferenceModel(
        ontology,
        item_collection,
        ratings,
        PreferenceModelVariant.SIP,
        historical_user_id="13",
    )
    interaction_model = InteractionModel(
        "data/interaction_models/cir6.yaml", annotated_conversations
    )
    nlu = IntentClassifierRasa()
    nlg = NLG()
    simulator = AgendaBasedSimulator(
        preference_model,
        interaction_model,
        nlu,
        nlg,
        ontology,
        item_collection,
        ratings,
    )
    simulate_conversation(agent, simulator)
