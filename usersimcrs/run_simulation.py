"""Console application for running simulation."""

import argparse
import configparser
import os

# import sys
import json
import yaml
from typing import Any

from dialoguekit.nlu.models.diet_classifier_rasa import IntentClassifierRasa
from dialoguekit.nlu.models.intent_classifier_cosine import (
    IntentClassifierCosine,
)
from dialoguekit.nlg.nlg import NLG
from dialoguekit.agent.agent import Agent
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.ontology import Ontology
from dialoguekit.core.recsys.item_collection import ItemCollection
from dialoguekit.core.recsys.ratings import Ratings
from dialoguekit.manager.dialogue_manager import DialogueManager
from dialoguekit.platforms.platform import Platform

# from dialoguekit.agent.terminal_agent import TerminalAgent
# from dialoguekit.agent.moviebot_agent import MovieBotAgent
from dialoguekit.core.intent import Intent
from dialoguekit.core.utterance import Utterance
from dialoguekit.nlu.models.satisfaction_classifier import (
    SatisfactionClassifier,
)

from usersimcrs.simulator.user_simulator import UserSimulator
from usersimcrs.sample_agent.sample_agent import SampleAgent
from usersimcrs.simulator.agenda_based_simulator import AgendaBasedSimulator
from usersimcrs.simulator.preference_model import (
    PreferenceModel,
    PreferenceModelVariant,
)
from usersimcrs.simulator.interaction_model import InteractionModel

# from usersimcrs.utils.persona_generator import Persona, Context


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


def parse_args() -> Any:
    conf_parser = argparse.ArgumentParser(
        description=__doc__,  # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
        add_help=False,
    )
    conf_parser.add_argument(
        "-c", "--conf_file", help="Specify config file", metavar="FILE"
    )
    args, remaining_argv = conf_parser.parse_known_args()
    default_args = {}
    if not os.path.exists(args.conf_file):
        raise FileNotFoundError(f"Config file not found: {args.conf_file}")
    else:
        cp = configparser.ConfigParser()
        cp.read([args.conf_file])
        default_args.update(dict(cp.items("SETTINGS")))

    parser = argparse.ArgumentParser()
    parser.set_defaults(**default_args)
    parser.add_argument(
        "-ontology", type=str, help="Path to ontology config file."
    )
    parser.add_argument(
        "-intents", type=str, help="Path to the intent scheme file."
    )
    parser.add_argument("-items", type=str, help="Path to item file.")
    parser.add_argument("-ratings", type=str, help="Path to rating file.")
    parser.add_argument(
        "-dialogues", type=str, help="Path to the annotated dialogues file."
    )
    parser.add_argument(
        "-satisfaction", type=str, help="Enables satisfaction classifier."
    )
    parser.add_argument(
        "-interaction_model",
        type=str,
        help="Interaction model to be used. Currently, either 'cosine' or "
        " 'diet'.",
    )
    args = parser.parse_args()

    # TODO Load settings from command line arguments or config file.
    if not os.path.exists(args.ontology):
        raise FileNotFoundError(f"Ontology file not found: {args.ontology}")
    if not os.path.exists(args.items):
        raise FileNotFoundError(f"Item file not found: {args.items}")
    if not os.path.exists(args.ratings):
        raise FileNotFoundError(f"Rating file not found: {args.ratings}")
    if not os.path.exists(args.dialogues):
        raise FileNotFoundError(
            f"Annotated dialogues file not found: {args.dialogues}"
        )
    if not os.path.exists(args.intents):
        raise FileNotFoundError(f"Intent schema file not found: {args.intents}")

    return args


def load_cosine_classifier(dialogues):
    gt_intents = []
    utterances = []
    for conversation in dialogues:
        for turn in conversation["conversation"]:
            if turn["participant"] == "AGENT":
                gt_intents.append(Intent(turn["intent"]))
                utterances.append(Utterance(turn["utterance"]))
    nlu = IntentClassifierCosine(intents=gt_intents)
    nlu.train_model(utterances=utterances, labels=gt_intents)
    return nlu


if __name__ == "__main__":
    agent = SampleAgent(agent_id="Tester")
    args = parse_args()

    ontology = Ontology(args.ontology)

    item_collection = ItemCollection()
    item_collection.load_items_csv(
        args.items, ["ID", "NAME", "genres", "keywords"]
    )

    ratings = Ratings(item_collection)
    ratings.load_ratings_csv(file_path=args.ratings)

    with open(args.dialogues) as annotated_dialogues_file:
        annotated_conversations = json.load(annotated_dialogues_file)
        interaction_model = InteractionModel(
            config_file=args.intents,
            annotated_conversations=annotated_conversations,
        )
        if args.interaction_model == "cosine":
            nlu = load_cosine_classifier(dialogues=annotated_conversations)

    satisfaction_model = None
    if args.satisfaction:
        satisfaction_model = SatisfactionClassifier()

    with open(args.intents) as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # TODO: initialization of the simulator with NLU, NLG, etc.
    preference_model = PreferenceModel(
        ontology,
        item_collection,
        ratings,
        PreferenceModelVariant.SIP,
        historical_user_id="13",
    )
    interaction_model = InteractionModel(args.intents, annotated_conversations)
    nlu = IntentClassifierRasa(
        config["agent_intents"],
        "data/agents/moviebot/annotated_dialogues_rasa_agent.yml",
        ".rasa",
    )
    nlg = NLG()
    nlg.template_from_file(template_file=args.dialogues)
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
