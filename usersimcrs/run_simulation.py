"""Console application for running simulation."""

import argparse
import json
import logging
import os
from typing import Set

import confuse
import requests
import yaml
from dialoguekit.connector.dialogue_connector import DialogueConnector
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.domain import Domain
from dialoguekit.core.intent import Intent
from dialoguekit.core.utterance import Utterance
from dialoguekit.nlg import ConditionalNLG
from dialoguekit.nlg.template_from_training_data import (
    extract_utterance_template,
)
from dialoguekit.nlu import NLU
from dialoguekit.nlu.intent_classifier import IntentClassifier
from dialoguekit.nlu.models.diet_classifier_rasa import IntentClassifierRasa
from dialoguekit.nlu.models.intent_classifier_cosine import (
    IntentClassifierCosine,
)
from dialoguekit.participant.agent import Agent
from dialoguekit.participant.participant import DialogueParticipant
from dialoguekit.platforms.platform import Platform
from sample_agents.moviebot_agent import MovieBotAgent

from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.items.ratings import Ratings
from usersimcrs.simulator.agenda_based.agenda_based_simulator import (
    AgendaBasedSimulator,
)
from usersimcrs.simulator.agenda_based.interaction_model import InteractionModel
from usersimcrs.simulator.user_simulator import UserSimulator
from usersimcrs.user_modeling.preference_model import PreferenceModel

DEFAULT_CONFIG_PATH = "config/default/config_default.yaml"
OUTPUT_DIR = "data/runs"

logging.basicConfig(
    format="[%(asctime)s] %(levelname)-12s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger()


def main(config: confuse.Configuration, agent: Agent) -> None:
    """Executes the specified configuration.

    Loads domain and interaction model. Initializes agent and user. Runs the
    simulation.

    Args:
        config: Configuration generated from YAML configuration file.
        agent: Conversational agent.

    Raises:
        TypeError: if the agent is not an instance of Agent.
    """
    if not isinstance(agent, Agent):
        raise TypeError(f"agent must be Agent, not {type(agent).__name__}")

    # Loads domain, item collection, and preference data
    domain = Domain(config["domain"].get())

    item_collection = ItemCollection()
    item_collection.load_items_csv(
        config["items"].get(),
        domain=domain,
        domain_mapping=config["domain_mapping"].get(),
        id_col=config["id_col"].get(),
    )

    ratings = Ratings(item_collection)
    ratings.load_ratings_csv(file_path=config["ratings"].get())
    historical_ratings, ground_truth_ratings = ratings.create_split(
        config["historical_ratings_ratio"].get(0.8)
    )

    preference_model = PreferenceModel(
        domain,
        item_collection,
        historical_ratings,
        historical_user_id="13",
    )

    # Loads dialogue sample
    annotated_dialogues_file = config["dialogues"].get()
    annotated_conversations = json.load(open(annotated_dialogues_file))

    # Loads interaction model
    interaction_model = InteractionModel(
        config_file=config["intents"].get(),
        annotated_conversations=annotated_conversations,
    )

    # NLU
    nlu = get_NLU(config)

    # NLG
    template = extract_utterance_template(
        annotated_dialogue_file=annotated_dialogues_file,
    )
    nlg = ConditionalNLG(template)

    simulator = AgendaBasedSimulator(
        "TEST03",
        preference_model,
        interaction_model,
        nlu,
        nlg,
        domain,
        item_collection,
        ratings,
    )
    simulate_conversation(agent, simulator)


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
    dc = DialogueConnector(agent, user_simulator, platform)
    dc.start()
    dc.close()
    return dc.dialogue_history


def parse_args() -> argparse.Namespace:
    """Defines accepted arguments and returns the parsed values.

    Returns:
        A namespace object containing the arguments.
    """

    parser = argparse.ArgumentParser(prog="run_simulation.py")
    parser.add_argument(
        "-c",
        "--config-file",
        help=(
            "Path to configuration file to overwrite default values. "
            "Defaults to None."
        ),
    )
    parser.add_argument(
        "-a",
        "--agent_id",
        type=str,
        help=("Id of the agent tested. Defaults to 'IAI MovieBot'."),
    )
    parser.add_argument(
        "--agent_uri",
        type=str,
        help="URI to communicate with the agent. By default we assume that the"
        " agent has an HTTP API.",
    )
    parser.add_argument(
        "-o",
        "--output_name",
        type=str,
        help="Specifies the output name for the simulation configuration.",
    )
    parser.add_argument(
        "--domain", type=str, help="Path to domain config file."
    )
    parser.add_argument(
        "--intents", type=str, help="Path to the intent schema file."
    )
    parser.add_argument("--items", type=str, help="Path to items file.")
    parser.add_argument(
        "--id_col", type=str, help="Name of the CSV field containing item id."
    )
    parser.add_argument(
        "--domain_mapping",
        type=json.loads,
        help="String form of field mapping.",
    )
    parser.add_argument("--ratings", type=str, help="Path to ratings file.")
    parser.add_argument(
        "--historical_ratings_ratio",
        type=float,
        help="Ratio of ratings to be used as historical data.",
    )
    parser.add_argument(
        "--dialogues", type=str, help="Path to the annotated dialogues file."
    )
    parser.add_argument(
        "--intent_classifier",
        choices=["cosine", "diet"],
        help="Intent classifier model to be used. Defaults to cosine.",
    )
    parser.add_argument(
        "--rasa_dialogues",
        type=str,
        help="Path to the Rasa annotated dialogues file.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_const",
        const=True,
        help=("Debug mode. Defaults to False."),
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> confuse.Configuration:
    """Loads config from config file and command line parameters.

    Loads default values from `config/default/config_default.yaml`. Values are
    then updated with any value specified in the command line arguments.

    Args:
        args: Arguments parsed with argparse.
    """
    # Load default config
    config = confuse.Configuration("usersimcrs")
    config.set_file(DEFAULT_CONFIG_PATH)

    # Load additional config (update defaults).
    if args.config_file:
        config.set_file(args.config_file)

    # Update config from command line arguments
    config.set_args(args, dots=True)

    # Save run config to metadata file
    output_name = config["output_name"].get()
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    output_config_file = os.path.join(OUTPUT_DIR, f"{output_name}.meta.yaml")
    with open(output_config_file, "w") as f:
        f.write(config.dump())

    return config


def get_NLU(config: confuse.Configuration) -> IntentClassifier:
    """Returns an NLU component.

    Only supports DialogueKit intent classifiers.

    Args:
        config: Configuration for the simulation.

    Raises:
        ValueError: Unsupported intent classifier.

    Returns:
        An NLU component.
    """

    intent_classifier = config["intent_classifier"].get()
    if intent_classifier == "cosine":
        # NLU without slot annotators
        return NLU(load_cosine_classifier(config), slot_annotators=[])
    elif intent_classifier == "diet":
        classifier = load_rasa_diet_classifier(config)
        return NLU(classifier, [classifier])
    elif intent_classifier:
        raise ValueError(
            "Unsupported intent classifier. Check DialogueKit intent"
            " classifiers."
        )


def load_cosine_classifier(
    config: confuse.Configuration,
) -> IntentClassifierCosine:
    """Trains a cosine classifier on annotated dialogues for NLU module.

    Args:
        config: Configuration generated from YAML configuration file.

    Returns:
        A trained cosine model for intent classification.
    """
    # TODO: Move to DialogueKit as util function.
    # See: https://github.com/iai-group/UserSimCRS/issues/92
    annotated_dialogues_file = config["dialogues"].get()
    dialogues = json.load(open(annotated_dialogues_file))

    gt_intents = []
    utterances = []
    for conversation in dialogues:
        for turn in conversation["conversation"]:
            if turn["participant"] == "AGENT":
                gt_intents.append(Intent(turn["intent"]))
                utterances.append(
                    Utterance(
                        turn["utterance"], participant=DialogueParticipant.AGENT
                    )
                )
    intent_classifier = IntentClassifierCosine(intents=gt_intents)
    intent_classifier.train_model(utterances=utterances, labels=gt_intents)
    return intent_classifier


def load_rasa_diet_classifier(
    config: confuse.Configuration,
) -> IntentClassifierCosine:
    """Trains a DIET classifier on Rasa annotated dialogues for NLU module.

    Args:
        config: Configuration generated from YAML configuration file.

    Returns:
        A trained Rasa DIET model for intent classification.
    """
    # TODO: Move to DialogueKit as util function.
    # See: https://github.com/iai-group/UserSimCRS/issues/92
    intent_schema_file = config["intents"].get()
    intent_schema = yaml.load(open(intent_schema_file), Loader=yaml.FullLoader)

    agent_intents_str: Set[str] = set()
    for v in intent_schema["user_intents"].values():
        if "expected_agent_intents" in v:
            agent_intents_str.update(v["expected_agent_intents"])
    # agent_intents_str = intent_schema["agent_elicit_intents"]
    # agent_intents_str.extend(intent_schema["agent_set_retrieval"])
    agent_intents = [Intent(intent) for intent in agent_intents_str]
    intent_classifier = IntentClassifierRasa(
        agent_intents,
        config["rasa_dialogues"].get(),
        ".rasa",
    )
    intent_classifier.train_model()
    return intent_classifier


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)

    if config["debug"].get():
        logger.setLevel(logging.DEBUG)

    # Defines the agent for the simulation.
    # By default, a local moviebot is used.
    # See usage example in README for more details.
    agent_uri = config["agent_uri"].get()
    try:
        response = requests.get(agent_uri)
        assert response.status_code == 200
        agent = MovieBotAgent(agent_id="MovieBotTester", uri=agent_uri)
    except requests.exceptions.RequestException:
        raise RuntimeError(
            f"Connection refused to {agent_uri}. Please check that "
            "the conversational agent is running at this address. See the full "
            "traceback above."
        )

    main(config, agent)
