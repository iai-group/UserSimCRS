"""Console application for running simulation."""

import argparse
import json
import os
import sys

import yaml
from dialoguekit.agent.agent import Agent
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.ontology import Ontology
from dialoguekit.manager.dialogue_manager import DialogueManager
from dialoguekit.nlg.nlg import NLG
from dialoguekit.nlu.models.diet_classifier_rasa import IntentClassifierRasa
from dialoguekit.platformss.platform import Platform

from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.items.ratings import Ratings
from usersimcrs.sample_agent.sample_agent import SampleAgent
from usersimcrs.simulator.agenda_based.agenda_based_simulator import (
    AgendaBasedSimulator,
)
from usersimcrs.simulator.agenda_based.interaction_model import InteractionModel
from usersimcrs.simulator.preference_model import (
    PreferenceModel,
    PreferenceModelVariant,
)
from usersimcrs.simulator.user_simulator import UserSimulator


def simulate_conversation(
    agent: Agent, user_simulator: UserSimulator
) -> Dialogue:
    """Simulates a single conversation and returns the resulting dialogue.

    Args:
        agent: An agent.
        user_simulator: A user simulator.
    """
    platform = Platform()  # TODO: Add simulator platform
    dm = DialogueManager(agent, user_simulator, platform)
    agent.connect_dialogue_manager(dialogue_manager=dm)
    user_simulator.connect_dialogue_manager(dialogue_manager=dm)
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
        sys.exit("FileNotFound: {file}".format(file=args.ontology))
    if not os.path.exists(args.items):
        sys.exit("FileNotFound: {file}".format(file=args.items))
    if not os.path.exists(args.ratings):
        sys.exit("FileNotFound: {file}".format(file=args.ratings))

    ontology = Ontology(args.ontology)

    item_collection = ItemCollection()
    item_collection.load_items_csv(args.items, ["ID", "NAME", "genres"])

    ratings = Ratings(item_collection)
    ratings.load_ratings_csv(args.ratings)

    annotated_dialogues_file = open(
        "data/agents/moviebot/annotated_dialogues.json"
    )
    annotated_conversations = json.load(annotated_dialogues_file)

    with open("data/interaction_models/cir6.yaml") as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)

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
    nlu = IntentClassifierRasa(
        config["agent_intents"],
        "data/agents/moviebot/annotated_dialogues_rasa_agent.yml",
        ".rasa",
    )
    nlg = NLG()
    nlg.template_from_file("data/agents/moviebot/annotated_dialogues.json")
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
