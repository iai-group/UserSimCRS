"""Script to optimize prompts and weights of extractor module."""

import argparse
import functools
import logging
import os
from statistics import mean
from typing import List

import confuse
import dspy
import yaml
from dspy.primitives.assertions import (
    assert_transform_module,
    backtrack_handler,
)
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2
from tqdm import tqdm

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.utils.dialogue_reader import json_to_dialogues
from usersimcrs.nlu.lm.lm_nlu import ExtractionModule
from usersimcrs.nlu.lm.optimization.metrics import (
    dialogue_acts_f1_score,
    dialogue_acts_precision,
    dialogue_acts_recall,
    intent_error_rate,
    slot_error_rate,
    validate_answer,
)

DEFAULT_CONFIG_PATH = "config/lm/config_optimization_default.yaml"


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Optimize prompts and weights."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to configuration file.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Path to output directory.",
    )
    parser.add_argument(
        "-p",
        "--participant",
        choices=["ALL", "USER", "AGENT"],
        help="Participant to optimize for.",
    )
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "-d",
        "--dialogues_path",
        type=str,
        help="Path to file containing annotated dialogues.",
    )
    data_group.add_argument(
        "-i",
        "--intent_labels_path",
        type=str,
        help="Path to file containing intent labels.",
    )
    data_group.add_argument(
        "-s",
        "--slot_labels_path",
        type=str,
        help="Path to file containing slot labels.",
    )
    lm_group = parser.add_argument_group("Language Model")
    lm_group.add_argument(
        "--ollama_base_url",
        type=str,
        help="Base URL of Ollama server.",
    )
    lm_group.add_argument(
        "--ollama_lm_model",
        type=str,
        help="Language model to use.",
    )
    miprov2_group = parser.add_argument_group("MIPROv2")
    miprov2_group.add_argument(
        "--num_candidates",
        type=int,
        dest="miprov2.num_candidates",
        help="Number of instructions and fewshot examples to generate.",
    )
    miprov2_group.add_argument(
        "--log_dir",
        type=str,
        dest="miprov2.log_dir",
        help="Path to log directory.",
    )
    miprov2_group.add_argument(
        "--init_temperature",
        type=float,
        dest="miprov2.init_temperature",
        help="Temperature for generating new instructions.",
    )
    compile_group = parser.add_argument_group("Compile arguments")
    compile_group.add_argument(
        "--num_batches",
        type=int,
        dest="compile_args.num_batches",
        help="Number of optimization trials to run.",
    )
    compile_group.add_argument(
        "--max_bootstrapped_demos",
        type=int,
        dest="compile_args.max_bootstrapped_demos",
        help="Maximum number of bootstrapped demonstrations.",
    )
    compile_group.add_argument(
        "--max_labeled_demos",
        type=int,
        dest="compile_args.max_labeled_demos",
        help="Maximum number of labeled demonstrations.",
    )
    eval_group = parser.add_argument_group("Evaluation arguments")
    eval_group.add_argument(
        "--num_threads",
        type=int,
        dest="evaluation_args.num_threads",
        help="Number of threads to use.",
    )
    eval_group.add_argument(
        "--display_progress",
        type=bool,
        dest="evaluation_args.display_progress",
        help="Whether to display progress.",
    )
    eval_group.add_argument(
        "--display_table",
        type=int,
        dest="evaluation_args.display_table",
        help="Whether to display table.",
    )
    return parser.parse_args()


def load_configuration(args: argparse.Namespace) -> confuse.Configuration:
    """Loads configuration from file and command-line arguments.

    Args:
        args: Command-line arguments.

    Returns:
        Configuration.
    """
    config = confuse.Configuration("LM Optimization")
    config.set_file(DEFAULT_CONFIG_PATH)

    if args.config:
        config.set_file(args.config)

    # Update configuration with command-line arguments
    config.set_args(args, dots=True)

    # Save configuration
    output_dir = config["output_dir"].get()
    os.makedirs(output_dir, exist_ok=True)

    with open(
        os.path.join(output_dir, "optimization_config.meta.yaml"), "w"
    ) as f:
        f.write(config.dump())

    return config


def filter_participant_utterances(
    dialogues: List[Dialogue], participant: str
) -> List[AnnotatedUtterance]:
    """Filters utterances of a given participant from dialogues.

    Participant can be "ALL", "USER", or "AGENT".

    Args:
        dialogues: List of dialogues.
        participant: Participant to load utterances for.

    Returns:
        List of utterances.
    """
    utterances = []
    for dialogue in dialogues:
        for utterance in dialogue.utterances:
            if participant == "ALL":
                utterances.append(utterance)
            elif utterance.participant.name == participant:
                utterances.append(utterance)
    return utterances


def create_example(utterance: AnnotatedUtterance) -> dspy.Example:
    """Creates an example from an utterance.

    Args:
        utterance: Annotated utterance.

    Returns:
        Example.
    """
    str_dialogue_acts = []
    for dialogue_act in utterance.dialogue_acts:
        slot_values = []
        for pair in dialogue_act.annotations:
            if pair.value:
                slot_values.append(f"{pair.slot}={pair.value}")
            else:
                slot_values.append(f"{pair.slot}")
        slot_values = ",".join(slot_values)
        str_dialogue_acts.append(f"{dialogue_act.intent.label}({slot_values})")

    return dspy.Example(
        input_utterance=utterance.text,
        dialogue_acts=str_dialogue_acts,
    ).with_inputs("input_utterance")


def create_examples(
    utterances: List[AnnotatedUtterance],
) -> List[dspy.Example]:
    """Creates examples from a list of utterances.

    Args:
        utterances: List of utterances.

    Returns:
        List of examples.
    """
    return [create_example(utterance) for utterance in utterances]


if __name__ == "__main__":
    args = parse_args()
    config = load_configuration(args)

    # Load intent and slot labels
    intent_labels = yaml.safe_load(open(config["intent_labels_path"].get(str)))
    slot_labels = yaml.safe_load(open(config["slot_labels_path"].get(str)))

    # Load utterances of participant from dialogues
    dialogues = json_to_dialogues(config["dialogues_path"].get(str))
    participant = config["participant"].get()
    utterances = filter_participant_utterances(dialogues, participant)

    # Load language model
    lm = dspy.OllamaLocal(
        model=config["ollama_lm_model"].get(str),
        base_url=config["ollama_base_url"].get(str),
    )
    dspy.settings.configure(lm=lm)

    dialogue_act_extractor = ExtractionModule(intent_labels, slot_labels)
    dialogue_act_extractor = assert_transform_module(
        dialogue_act_extractor,
        functools.partial(backtrack_handler, max_backtracks=3),
    )

    # Prepare data
    examples = create_examples(utterances)

    # Slice data
    train_examples = examples[: int(0.8 * len(examples))]
    val_examples = examples[int(0.8 * len(examples)) :]

    # Optimization using MIPROv2
    os.makedirs(config["miprov2"]["log_dir"].get(), exist_ok=True)
    miprov2_kwargs = {k: v.get() for k, v in config["miprov2"].items()}
    optimizer = MIPROv2(
        metric=validate_answer,
        prompt_model=lm,
        task_model=lm,
        **miprov2_kwargs,
    )
    compile_kwargs = {k: v.get() for k, v in config["compile_args"].items()}
    eval_kwargs = {k: v.get() for k, v in config["evaluation_args"].items()}
    optimized_program = optimizer.compile(
        dialogue_act_extractor,
        trainset=train_examples,
        requires_permission_to_run=False,
        eval_kwargs=eval_kwargs,
        **compile_kwargs,
    )

    # Save optimized program
    output_dir = config["output_dir"].get()
    optimized_program.save(os.path.join(output_dir, "optimized_program.json"))

    # Evaluate optimized program
    scores = {
        "SER": [],
        "IER": [],
        "DAR": [],
        "DAP": [],
        "DAF1": [],
    }
    for val_example in tqdm(val_examples):
        try:
            prediction = dialogue_act_extractor(val_example.input_utterance)
            target_dialogue_acts = ExtractionModule.parse_dialogue_acts(
                val_example.dialogue_acts
            )

            predicted_dialogue_acts = ExtractionModule.parse_dialogue_acts(
                prediction.dialogue_acts
            )

            scores["SER"].append(
                slot_error_rate(predicted_dialogue_acts, target_dialogue_acts)
            )
            scores["IER"].append(
                intent_error_rate(
                    predicted_dialogue_acts, target_dialogue_acts
                )
            )
            scores["DAR"].append(
                dialogue_acts_recall(
                    predicted_dialogue_acts,
                    target_dialogue_acts,
                )
            )
            scores["DAP"].append(
                dialogue_acts_precision(
                    predicted_dialogue_acts,
                    target_dialogue_acts,
                )
            )
            scores["DAF1"].append(
                dialogue_acts_f1_score(
                    predicted_dialogue_acts,
                    target_dialogue_acts,
                )
            )
        except Exception as e:
            logging.error(f"Error: {e}")

    logging.info(
        "Evaluation scores:\n"
        f"{[f'{k}: {round(mean(v),3)}' for k, v in scores.items()]}"
    )
