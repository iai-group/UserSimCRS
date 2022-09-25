"""Console application for running simulation."""

import argparse
import json
import os
from typing import Any, Dict

import yaml
from dialoguekit.agent.agent import Agent
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.intent import Intent
from dialoguekit.core.ontology import Ontology
from dialoguekit.core.recsys.item_collection import ItemCollection
from dialoguekit.core.recsys.ratings import Ratings
from dialoguekit.core.utterance import Utterance
from dialoguekit.manager.dialogue_manager import DialogueManager
from dialoguekit.nlg.nlg import NLG
from dialoguekit.nlu.models.diet_classifier_rasa import IntentClassifierRasa
from dialoguekit.nlu.models.intent_classifier_cosine import (
    IntentClassifierCosine,
)
from dialoguekit.nlu.models.satisfaction_classifier import (
    SatisfactionClassifier,
)
from dialoguekit.platforms.platform import Platform

from usersimcrs.sample_agent.sample_agent import SampleAgent
from usersimcrs.simulator.agenda_based.agenda_based_simulator import (
    AgendaBasedSimulator,
)
from usersimcrs.simulator.agenda_based.interaction_model import InteractionModel
from usersimcrs.simulator.user_simulator import UserSimulator
from usersimcrs.user_modeling.preference_model import (
    PreferenceModel,
    PreferenceModelVariant,
)


def parse_args() -> Any:
    """Parses arguments in a .ini file and/or via the command line. The .ini
        config file is used to set default values which can be overridden via
        the command line.

    Returns:
        A namespace object containing the arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config", help="Specify config file *.yaml", metavar="FILE"
    )
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
        "diet"
        ".",
    )
    args = parser.parse_args()
    print("arguments: {}".format(str(args)))

    opt = vars(args)
    print("opt", opt)
    args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    print("Yargs", args)
    opt.update(args)
    args = opt
    print("arguments: {}".format(str(args)))

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


def load_cosine_classifier(
    dialogues: Dict[str, Dict[str, str]]
) -> IntentClassifierCosine:
    """Trains a cosine classifier on annotated dialogues for NLU module.

    Args:
        dialogues: A JSON format of annotated dialogues.

    Returns:
        A trained cosine model for intent classification.
    """
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


def simulate_conversation(
    agent: Agent, user_simulator: UserSimulator
) -> Dialogue:
    """Simulates a single conversation and returns the resulting dialogue.

    Args:
        agent: An agent.
        user_simulator: A user simulator.

    Returns:
        The simulated dialogue.
    """
    platform = Platform()  # TODO: Add simulator platform
    dm = DialogueManager(agent, user_simulator, platform)
    agent.connect_dialogue_manager(dialogue_manager=dm)
    user_simulator.connect_dialogue_manager(dialogue_manager=dm)
    dm.start()
    dm.close()
    return dm.dialogue_history


if __name__ == "__main__":
    agent = SampleAgent(agent_id="Tester")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config", help="Specify config file *.yaml", metavar="FILE"
    )
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
        "diet"
        ".",
    )
    args = parser.parse_args()
    print("arguments: {}".format(str(args)))

    opt = vars(args)
    print("opt", opt)
    args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    print("Yargs", args)
    opt.update(args)
    args = opt
    print("arguments: {}".format(str(args)))
    print("arguments2: ", args)

    ontology = Ontology(args["ontology"])

    item_collection = ItemCollection()
    item_collection.load_items_csv(
        args["items"], ["ID", "NAME", "genres", "keywords"]
    )

    ratings = Ratings(item_collection)
    ratings.load_ratings_csv(file_path=args["ratings"])

    with open(args["dialogues"]) as annotated_dialogues_file:
        annotated_conversations = json.load(annotated_dialogues_file)
        interaction_model = InteractionModel(
            config_file=args["intents"],
            annotated_conversations=annotated_conversations,
        )
        if args["interaction_model"] == "cosine":
            nlu = load_cosine_classifier(dialogues=annotated_conversations)

    satisfaction_model = None
    if args["satisfaction"]:
        satisfaction_model = SatisfactionClassifier()

    with open(args["intents"]) as yaml_file:
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
        args["intents"], annotated_conversations
    )
    agent_intents_str = config["agent_elicit_intents"]
    agent_intents_str.extend(config["agent_set_retrieval"])
    agent_intents = [Intent(intent) for intent in agent_intents_str]
    print("Agent intents", agent_intents)
    nlu = IntentClassifierRasa(
        agent_intents,
        "data/agents/moviebot/annotated_dialogues_rasa_agent.yml",
        ".rasa",
    )
    nlg = NLG()
    nlg.template_from_file(template_file=args["dialogues"])
    simulator = AgendaBasedSimulator(
        "TEST03",
        preference_model,
        interaction_model,
        nlu,
        nlg,
        ontology,
        # item_collection,
        # ratings,
    )
    simulate_conversation(agent, simulator)
    print("FINISHED")
