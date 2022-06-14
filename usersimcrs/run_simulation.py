"""Console application for running simulation."""

import argparse
import configparser
import os
import sys
import json
import yaml
import datetime

from dialoguekit.nlu.models.diet_classifier_rasa import IntentClassifierRasa
from dialoguekit.nlu.models.intent_classifier_cosine import IntentClassifierCosine
from dialoguekit.nlg.nlg import NLG
from dialoguekit.nlu.nlu import NLU
from dialoguekit.agent.agent import Agent
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.ontology import Ontology
from dialoguekit.core.recsys.item_collection import ItemCollection
from dialoguekit.core.recsys.ratings import Ratings
from dialoguekit.manager.dialogue_manager import DialogueManager
from dialoguekit.platforms.platform import Platform
from dialoguekit.agent.terminal_agent import TerminalAgent
from dialoguekit.agent.moviebot_agent import MovieBotAgent
from dialoguekit.core.intent import Intent
from dialoguekit.core.utterance import Utterance
from dialoguekit.nlu.models.satisfaction_classifier import SatisfactionClassifier

from usersimcrs.simulator.user_simulator import UserSimulator
from usersimcrs.sample_agent.sample_agent import SampleAgent
from usersimcrs.simulator.agenda_based_simulator import AgendaBasedSimulator
from usersimcrs.simulator.preference_model import (
    PreferenceModel,
    PreferenceModelVariant,
)
from usersimcrs.simulator.interaction_model import InteractionModel
from usersimcrs.utils.persona_generator import Persona, Context


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
    # agent = SampleAgent(agent_id="sample_agent")
    # agent = TerminalAgent("test")
    conf_parser = argparse.ArgumentParser(
        description=__doc__, # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
        add_help=False
        )
    conf_parser.add_argument("-c", "--conf_file",
                        help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()
    default_args = {}
    if args.conf_file:
        if not os.path.exists(args.conf_file):
            sys.exit("FileNotFound: {file}".format(file=args.conf_file))
        else:
            cp = configparser.ConfigParser()
            cp.read([args.conf_file])
            default_args.update(dict(cp.items("SETTINGS")))

    
    parser = argparse.ArgumentParser(parents=[conf_parser])
    parser.set_defaults(**default_args)
    parser.add_argument(
        "-ontology", type=str, help="Path to the ontology config file."
    )
    parser.add_argument("-items", type=str, help="Path to the items file.")
    parser.add_argument("-ratings", type=str, help="Path to the ratings file.")
    parser.add_argument(
        "-dialogues", type=str, help="Path to the annotated dialogues file."
        )
    parser.add_argument(
        "-ir", type=str, help="Path to the intent scheme file."
        )
    args = parser.parse_args(remaining_argv)

    # TODO Load settings from command line arguments or config file.
    if not os.path.exists(args.ontology):
        sys.exit("FileNotFound: {file}".format(file=args.ontology))
    if not os.path.exists(args.items):
        sys.exit("FileNotFound: {file}".format(file=args.items))
    if not os.path.exists(args.ratings):
        sys.exit("FileNotFound: {file}".format(file=args.ratings))
    if not os.path.exists(args.dialogues):
        sys.exit("FileNotFound: {file}".format(file=args.dialogues))
    if not os.path.exists(args.ir):
        sys.exit("FileNotFound: {file}".format(file=args.ir))

    ontology = Ontology(args.ontology)

    item_collection = ItemCollection()
    item_collection.load_items_csv(args.items, ["ID", "NAME", "genres","keywords"])

    ratings = Ratings(item_collection)
    ratings.load_ratings_csv(args.ratings)

    with open(
        args.dialogues
    ) as annotated_dialogues_file:
        annotated_conversations = json.load(annotated_dialogues_file)
    gt_intents = []
    utterances = []
    for conversation in annotated_conversations:
        for turn in conversation["conversation"]:
            if turn["participant"] == "AGENT":
                gt_intents.append(Intent(turn["intent"]))
                utterances.append(Utterance(turn["utterance"]))

    with open("data/interaction_models/cir6_v2.yaml") as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    agent_intents = [Intent(x) for x in config["agent_intents"]]

    # TODO: initialization of the simulator with NLU, NLG, etc.
    preference_model = PreferenceModel(
        ontology,
        item_collection,
        ratings,
        PreferenceModelVariant.PKG,
        historical_user_id="1",
    )
    interaction_model = InteractionModel(
        args.ir, annotated_conversations
    )
    satisfaction_model = SatisfactionClassifier()
    nluc = IntentClassifierCosine(agent_intents)
    nluc.train_model(utterances,gt_intents)

    nlur = IntentClassifierRasa(
        agent_intents,
        "data\\agents\\moviebot\\annotated_dialogues_v2_rasa_agent.yaml",
        ".rasa",
    )
    ctx = Context(group_setting=False, time_of_the_day=(datetime.time(21,0,0),datetime.time(23,59,59)),weekend=False)
    persona = Persona("Jafar","1",3.0,0.5,ctx)
    persona.calculate_max_retries()
    nlg = NLG()
    nlg.template_from_file(args.dialogues,"USER",satisfaction_model)
    nlur.train_model()
    nlu = NLU(nluc,[nlur])

    simulator = AgendaBasedSimulator(
        "simulator",
        preference_model,
        interaction_model,
        nlu,
        nlg,
        ontology,
        item_collection,
        ratings,
        persona,
        satisfaction_model
    )
    agent = MovieBotAgent("14","http://152.94.232.28:5001")
    simulate_conversation(agent, simulator)