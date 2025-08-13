"""Utility functions to run the simulation."""

import json
from typing import Any, Dict, Set, Tuple, Type

import confuse
import yaml
from dialoguekit.core.intent import Intent
from dialoguekit.core.utterance import Utterance
from dialoguekit.nlg import ConditionalNLG
from dialoguekit.nlg.nlg_abstract import AbstractNLG
from dialoguekit.nlg.template_from_training_data import extract_utterance_template
from dialoguekit.nlu import NLU
from dialoguekit.nlu.disjoint_dialogue_act_extractor import DisjointDialogueActExtractor
from dialoguekit.nlu.intent_classifier import IntentClassifier
from dialoguekit.nlu.models.diet_classifier_rasa import IntentClassifierRasa
from dialoguekit.nlu.models.intent_classifier_cosine import IntentClassifierCosine
from dialoguekit.participant import Agent
from dialoguekit.participant.participant import DialogueParticipant
from dialoguekit.utils.dialogue_reader import json_to_dialogues

from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.items.ratings import Ratings
from usersimcrs.nlu.lm.lm_dialogue_act_extractor import LMDialogueActsExtractor
from usersimcrs.simulator.agenda_based.interaction_model import InteractionModel
from usersimcrs.user_modeling.persona import Persona
from usersimcrs.user_modeling.simple_preference_model import SimplePreferenceModel


def map_path_to_class(cls_path: str) -> Type:
    """Maps a class path to a class.

    Args:
        cls_path: Class path.

    Returns:
        Class.
    """
    parts = cls_path.split(".")
    module_path = ".".join(parts[:-1])
    class_name = parts[-1]
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def get_agent_information(
    config: confuse.Configuration,
) -> Tuple[Type, Dict[str, Any]]:
    """Gets the agent information.

    Args:
        config: Configuration of the run.

    Raises:
        TypeError: if agent class does not inherit from Agent.

    Returns:
        Agent class and agent configuration.
    """
    agent_class = map_path_to_class(config["agent_class_path"].get())
    agent_config = {
        "id": config["agent_id"].get(),
        "uri": config["agent_uri"].get(),
    }

    if "agent_config" in config:
        agent_config.update(config["agent_config"].get())

    if not issubclass(agent_class, Agent):
        raise TypeError(f"Agent class {agent_class} must inherit from Agent.")

    return agent_class, agent_config


def get_simulator_information(
    config: confuse.Configuration,
) -> Tuple[str, Type, Dict[str, Any]]:
    """Gets the simulator information.

    Args:
        config: Configuration of the run.

    Raises:
        ValueError: If the simulator class is not supported.

    Returns:
        Simulator ID, simulator class, and configuration.
    """
    simulator_id = config["simulator_id"].get()
    simulator_class = map_path_to_class(config["simulator_class_path"].get())
    simulator_config = {}

    if simulator_class.__name__ == "AgendaBasedSimulator":
        simulator_config.update(_get_agenda_based_simulator_config(config))
    elif simulator_class.__name__ == "SinglePromptUserSimulator":
        simulator_config.update(
            _get_single_prompt_user_simulator_config(config)
        )
    elif simulator_class.__name__ == "DualPromptUserSimulator":
        simulator_config.update(_get_dual_prompt_user_simulator_config(config))
    else:
        raise ValueError(f"Simulator class {simulator_class} is not supported.")
    return simulator_id, simulator_class, simulator_config


def _get_agenda_based_simulator_config(
    config: confuse.Configuration,
) -> Dict[str, Any]:
    """Gets the configuration of the agenda-based simulator.

    Args:
        config: Configuration of the run.

    Returns:
        Configuration of the agenda-based simulator.
    """
    # Loads domain, item collection, and preference data
    domain = SimulationDomain(config["domain"].get())

    item_collection = ItemCollection(
        config["collection_db_path"].get(), config["collection_name"].get()
    )
    if config["items"].get() is not None:
        # Load items if CSV file is provided
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

    preference_model = SimplePreferenceModel(
        domain,
        item_collection,
        historical_ratings,
        historical_user_id="13",
    )

    # Loads dialogue sample
    annotated_dialogues_file = config["dialogues"].get()
    annotated_conversations = json_to_dialogues(
        annotated_dialogues_file,
        agent_ids=[config["agent_id"].get()],
        user_ids=["User"],
    )

    # Loads interaction model
    interaction_model = InteractionModel(
        config_file=config["intents"].get(),
        domain=domain,
        annotated_conversations=annotated_conversations,
    )

    # NLU
    nlu = get_NLU(config)

    # NLG
    nlg = get_NLG(config)

    return {
        "preference_model": preference_model,
        "interaction_model": interaction_model,
        "nlu": nlu,
        "nlg": nlg,
        "domain": domain,
        "item_collection": item_collection,
        "ratings": ratings,
    }


def get_NLU(config: confuse.Configuration) -> NLU:
    """Returns an NLU component.

    Args:
        config: Configuration for the simulation.

    Raises:
        ValueError: Unsupported intent classifier.

    Returns:
        An NLU component.
    """

    intent_classifier = config["intent_classifier"].get()
    classifier: IntentClassifier = None
    if intent_classifier == "cosine":
        # NLU without slot annotators
        classifier = train_cosine_classifier(config)
        return NLU(
            DisjointDialogueActExtractor(classifier, slot_value_annotators=[])
        )
    elif intent_classifier == "diet":
        classifier = train_rasa_diet_classifier(config)
        return NLU(DisjointDialogueActExtractor(classifier, [classifier]))
    elif intent_classifier == "lm":
        return LMDialogueActsExtractor(config["intent_classifier_config"].get())
    raise ValueError(
        "Unsupported intent classifier. Check DialogueKit intent"
        " classifiers."
    )


def train_cosine_classifier(
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
                gt_intents.extend(
                    [Intent(da["intent"]) for da in turn["dialogue_acts"]]
                )
                utterances.append(
                    Utterance(
                        turn["utterance"],
                        participant=DialogueParticipant.AGENT,
                    )
                )
    intent_classifier = IntentClassifierCosine(intents=gt_intents)
    intent_classifier.train_model(utterances=utterances, labels=gt_intents)
    return intent_classifier


def train_rasa_diet_classifier(
    config: confuse.Configuration,
) -> IntentClassifierRasa:
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
        intents = v.get("expected_agent_intents", []) or []
        agent_intents_str.update(intents)
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


def get_NLG(config: confuse.Configuration) -> AbstractNLG:
    """Returns an NLG component.

    Args:
        config: Configuration for the simulation.

    Raises:
        ValueError: Unsupported NLG component.

    Returns:
        An NLG component.
    """
    nlg_type = config["nlg"].get()
    if nlg_type == "conditional":
        annotated_dialogues_file = config["dialogues"].get()
        template = extract_utterance_template(
            annotated_dialogue_file=annotated_dialogues_file,
        )
        return ConditionalNLG(template)
    elif nlg_type == "lm":
        return map_path_to_class(config["nlg_class_path"].get())(
            **config["nlg_args"].get()
        )
    raise ValueError("Unsupported NLG component.")


def _get_single_prompt_user_simulator_config(
    config: confuse.Configuration,
) -> Dict[str, Any]:
    """Gets the configuration of the single prompt user simulator.

    Args:
        config: Configuration of the run.

    Returns:
        Configuration of the single prompt user simulator.
    """
    domain = SimulationDomain(config["domain"].get())
    item_type = config["item_type"].get()

    item_collection = ItemCollection(
        config["collection_db_path"].get(), config["collection_name"].get()
    )
    if config["items"].get() is not None:
        # Load items if CSV file is provided
        item_collection.load_items_csv(
            config["items"].get(),
            domain=domain,
            domain_mapping=config["domain_mapping"].get(),
            id_col=config["id_col"].get(),
        )

    llm_interface_class = map_path_to_class(
        config["llm_interface_class_path"].get()
    )
    llm_interface_args = config["llm_interface_args"].get()
    llm_interface = llm_interface_class(**llm_interface_args)

    task_definition = config["task_definition"].get()

    persona = None
    if "persona" in config:
        persona = Persona(config["persona"].get())

    return {
        "domain": domain,
        "item_collection": item_collection,
        "llm_interface": llm_interface,
        "item_type": item_type,
        "task_definition": task_definition,
        "persona": persona,
    }


def _get_dual_prompt_user_simulator_config(
    config: confuse.Configuration,
) -> Dict[str, Any]:
    """Gets the configuration of the dual prompt user simulator.

    Args:
        config: Configuration of the run.

    Returns:
        Configuration of the dual prompt user simulator.
    """
    simulator_config = _get_single_prompt_user_simulator_config(config)
    simulator_config["stop_definition"] = config["stop_definition"].get()
    return simulator_config
