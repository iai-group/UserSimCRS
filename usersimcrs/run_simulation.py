"""Console application for running simulation."""

import argparse
import json
import os
import sys

import numpy as np
import yaml
from dialoguekit.agent.agent import Agent
from dialoguekit.connector.dialogue_connector import DialogueConnector
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.domain import Domain
from dialoguekit.core.intent import Intent
from dialoguekit.core.recsys.item_collection import ItemCollection
from dialoguekit.core.recsys.ratings import Ratings
from dialoguekit.nlg import ConditionalNLG
from dialoguekit.nlg.template_from_training_data import (
    extract_utterance_template,
)
from dialoguekit.nlu import NLU, SatisfactionClassifierSVM
from dialoguekit.nlu.models.diet_classifier_rasa import IntentClassifierRasa
from dialoguekit.platforms.platform import Platform

from usersimcrs.sample_agent.mdp_agent import MDPAgent
from usersimcrs.simulator.agenda_based.agenda_based_simulator import (
    AgendaBasedSimulator,
)
from usersimcrs.simulator.agenda_based.interaction_model import InteractionModel
from usersimcrs.simulator.user_simulator import UserSimulator
from usersimcrs.user_modeling.preference_model import (
    PreferenceModel,
    PreferenceModelVariant,
)


def simulate_conversation(
    agent: Agent, user_simulator: UserSimulator
) -> Dialogue:
    """Simulates a single conversation and returns the resulting dialogue.

    Args:
        agent: An agent.
        user_simulator: A user simulator.
    """
    platform = Platform()  # TODO: Add simulator platform
    dm = DialogueConnector(
        agent, user_simulator, platform, save_dialogue_history=False
    )
    agent.connect_dialogue_connector(dialogue_connector=dm)
    user_simulator.connect_dialogue_connector(dialogue_connector=dm)
    dm.start()
    dm.close()
    return dm.dialogue_history


def parse_cmdline_arguments() -> argparse.Namespace:
    """Defines accepted arguments and returns the parsed values.

    Returns:
        Object with a property for each argument.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-domain", type=str, help="Path to domain config file.")
    parser.add_argument("-items", type=str, help="Path to items file.")
    parser.add_argument("-ratings", type=str, help="Path to ratings file.")

    return parser.parse_args()


def file_exist(filepath: str) -> None:
    """Tests if a file exists.

    Raises:
        SystemExit if file does not exit.
    """
    if not os.path.exists(filepath):
        sys.exit("FileNotFound: {file}".format(file=filepath))


def initialize_user_simulator(
    args: argparse.Namespace,
) -> UserSimulator:
    """Initializes the user simulator.

    Args:
        args: command line arguments.

    Returns:
        A user simulator.
    """
    # 1. Domain and item collection
    domain = Domain(args.domain)

    item_collection = ItemCollection()
    item_collection.load_items_csv(
        args.items, ["ID", "NAME", "genres", "keywords"]
    )

    # 2. Preference data
    ratings = Ratings(item_collection)
    ratings.load_ratings_csv(args.ratings)

    # 4. Annotated sample
    annotated_dialogue_path = "data/agents/MDP/annotated_dialogues_QRFA.json"
    annotated_dialogue_file = open(annotated_dialogue_path)
    annotated_conversations = json.load(annotated_dialogue_file)

    # 5. Load interaction model
    interaction_model_path = "data/interaction_models/qrfa.yaml"
    interaction_model = InteractionModel(
        interaction_model_path, annotated_conversations
    )
    InteractionModel.START_INTENT = Intent("QUERY")

    # 6. Define user model / population
    preference_model = PreferenceModel(
        domain,
        item_collection,
        ratings,
        PreferenceModelVariant.SIP,
        historical_user_id="13",
    )

    # 7. Train NLU components
    nlu_training_data_path = (
        "data/agents/MDP/annotated_dialogues_QRFA_rasa_agent.yaml"
    )
    with open(interaction_model_path) as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    agent_intents = [Intent(i) for i in config["agent_intents"].keys()]
    intent_classifier = IntentClassifierRasa(
        agent_intents,
        nlu_training_data_path,
        ".rasa",
    )
    intent_classifier.train_model()
    nlu = NLU(
        intent_classifier,
        [intent_classifier],
    )
    # 7. Train NLG component
    template = extract_utterance_template(
        annotated_dialogue_file=annotated_dialogue_path,
        satisfaction_classifier=SatisfactionClassifierSVM(),
    )
    nlg = ConditionalNLG(response_templates=template)

    simulator = AgendaBasedSimulator(
        "simulator_agent",
        preference_model,
        interaction_model,
        nlu,
        nlg,
        domain,
        item_collection,
        ratings,
    )
    return simulator


def initialize_agent() -> Agent:
    """Initializes the CRS agent.

    Here you can initialize the CRS agent of your choice, note the agent needs
    to inherit from the Agent class in dialoguekit. An example creating a MDP
    agent is provided.

    Returns:
        An agent.
    """
    agent = create_sample_mdp_agent()
    return agent


def create_sample_mdp_agent() -> MDPAgent:
    """Creates a sample MDP agent."""
    stop_state = Intent("EXIT")
    actions = ["reply"]
    states = [
        Intent("QUERY"),
        Intent("FEEDBACK"),
        Intent("REQUEST"),
        Intent("ANSWER"),
    ]
    transition_model = np.zeros((len(states), len(actions), len(states)))
    transition_model[0][0] = [0, 0, 0.7, 0.3]
    transition_model[1][0] = [0, 0, 0.2, 0.8]
    policy = [0, 0, 0, 0]

    # Train NLU component
    interaction_model_path = "data/interaction_models/qrfa.yaml"
    nlu_training_data_path = (
        "data/agents/MDP/annotated_dialogues_QRFA_rasa_user.yaml"
    )
    with open(interaction_model_path) as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    user_intents = [Intent(i) for i in config["user_intents"].keys()]
    intent_classifier = IntentClassifierRasa(
        user_intents,
        nlu_training_data_path,
        ".rasa",
    )
    intent_classifier.train_model()
    nlu = NLU(
        intent_classifier,
        [intent_classifier],
    )

    # Train NLG component
    annotated_dialogue_path = "data/agents/MDP/annotated_dialogues_QRFA.json"
    template = extract_utterance_template(
        annotated_dialogue_file=annotated_dialogue_path,
        satisfaction_classifier=SatisfactionClassifierSVM(),
        participant_to_learn="AGENT",
    )
    nlg = ConditionalNLG(response_templates=template)

    agent = MDPAgent(
        "mdp_agent",
        nlu,
        nlg,
        states,
        stop_state,
        actions,
        transition_model,
        policy,
    )

    return agent


if __name__ == "__main__":

    args = parse_cmdline_arguments()

    # TODO Load settings from command line arguments or config file.
    file_exist(args.domain)
    file_exist(args.items)
    file_exist(args.ratings)

    simulator = initialize_user_simulator(args)
    agent = initialize_agent()

    dialog_history = simulate_conversation(agent, simulator)
