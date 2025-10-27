"""Utility functions to run the simulation."""

import json
from typing import Any, Dict, Tuple, Type

import confuse
from dialoguekit.core.intent import Intent
from dialoguekit.core.utterance import Utterance
from dialoguekit.nlg import ConditionalNLG
from dialoguekit.nlg.nlg_abstract import AbstractNLG
from dialoguekit.nlg.template_from_training_data import (
    extract_utterance_template,
)
from dialoguekit.nlu import NLU
from dialoguekit.nlu.disjoint_dialogue_act_extractor import (
    DisjointDialogueActExtractor,
)
from dialoguekit.nlu.models.intent_classifier_cosine import (
    IntentClassifierCosine,
)
from dialoguekit.participant import Agent
from dialoguekit.participant.participant import DialogueParticipant
from dialoguekit.utils.dialogue_reader import json_to_dialogues

from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.items.ratings import Ratings
from usersimcrs.nlu.llm.llm_dialogue_act_extractor import (
    LLMDialogueActsExtractor,
)
from usersimcrs.simulator.agenda_based.interaction_model import InteractionModel
from usersimcrs.simulator.llm.interfaces.llm_interface import LLMInterface
from usersimcrs.user_modeling.persona import Persona
from usersimcrs.user_modeling.simple_preference_model import (
    SimplePreferenceModel,
)


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
    elif simulator_class.__name__ == "LLMSinglePromptUserSimulator":
        simulator_config.update(
            _get_llm_single_prompt_user_simulator_config(config)
        )
    elif simulator_class.__name__ == "LLMDualPromptUserSimulator":
        simulator_config.update(
            _get_llm_dual_prompt_user_simulator_config(config)
        )
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
    historical_ratings, _ = ratings.create_split(
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
    nlu_config = config["nlu"].get()
    intent_classifier = nlu_config.get("intent_classifier")
    if intent_classifier == "cosine":
        # NLU without slot annotators
        classifier = train_cosine_classifier(config)
        return NLU(
            DisjointDialogueActExtractor(classifier, slot_value_annotators=[])
        )
    elif intent_classifier == "llm":
        nlu_llm_interface = _get_llm_interface(nlu_config)
        return LLMDialogueActsExtractor(
            nlu_llm_interface, nlu_config.get("intent_classifier_config")
        )
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


def get_NLG(config: confuse.Configuration) -> AbstractNLG:
    """Returns an NLG component.

    Args:
        config: Configuration for the simulation.

    Raises:
        ValueError: Unsupported NLG component.

    Returns:
        An NLG component.
    """
    nlg_config = config["nlg"].get()
    nlg_type = nlg_config.get("type")
    if nlg_type == "conditional":
        annotated_dialogues_file = config["dialogues"].get()
        template = extract_utterance_template(
            annotated_dialogue_file=annotated_dialogues_file,
        )
        return ConditionalNLG(template)
    elif nlg_type == "llm":
        nlg_llm_interface = _get_llm_interface(nlg_config)
        nlg_args = nlg_config.get("args", {})
        nlg_args["llm_interface"] = nlg_llm_interface
        return map_path_to_class(nlg_config.get("class_path"))(**nlg_args)
    raise ValueError("Unsupported NLG component.")


def _get_llm_single_prompt_user_simulator_config(
    config: confuse.Configuration,
) -> Dict[str, Any]:
    """Gets the configuration of the LLM single-prompt user simulator.

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

    llm_interface = _get_llm_interface(dict(config))

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


def _get_llm_interface(config: Dict[str, Any]) -> LLMInterface:
    """Returns the LLM interface.

    Args:
        config: Configuration.

    Returns:
        LLM interface.
    """
    llm_interface_class = map_path_to_class(
        config.get("llm_interface_class_path")
    )
    llm_interface_args = config.get("llm_interface_args", {})
    llm_interface = llm_interface_class(**llm_interface_args)
    return llm_interface


def _get_llm_dual_prompt_user_simulator_config(
    config: confuse.Configuration,
) -> Dict[str, Any]:
    """Gets the configuration of the LLM dual-prompt user simulator.

    Args:
        config: Configuration of the run.

    Returns:
        Configuration of the dual prompt user simulator.
    """
    simulator_config = _get_llm_single_prompt_user_simulator_config(config)
    simulator_config["stop_definition"] = config["stop_definition"].get()
    return simulator_config
