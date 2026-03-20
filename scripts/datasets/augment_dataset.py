"""Augment the formatted dataset with additional information.

The augmentation is artificial and done to create training data for neural user
simulators. Each dialogue is augmented with an information need and each
utterance is augmented with dialogue acts. The information need is inferred from
the dialogue acts, e.g., the annotations of a inquire dialogue act can serve as
requests.
"""

import argparse
import json
import yaml
from tqdm import tqdm

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.nlu.nlu import NLU
from dialoguekit.participant.participant import DialogueParticipant
from dialoguekit.utils.dialogue_reader import json_to_dialogues
from scripts.datasets.information_need_annotation.information_need_annotator import (  # noqa: E501
    DEFAULT_INITIAL_PROMPT_MOVIES_FILE,
    InformationNeedAnnotator,
)
from scripts.nlu.dialogue_acts_extraction import get_dialogue_acts_extractor
from usersimcrs.utils.simulation_utils import get_llm_interface


def annotate_dialogue_acts(
    user_nlu: NLU, agent_nlu: NLU, dialogue: Dialogue
) -> Dialogue:
    """Annotates the dialogue acts in the dialogue.

    Args:
        user_nlu: NLU to use for user dialogue act annotation.
        agent_nlu: NLU to use for agent dialogue act annotation.
        dialogue: Dialogue to annotate.

    Returns:
        Dialogue with annotated dialogue acts.
    """
    for utterance in dialogue.utterances:
        if utterance.participant == DialogueParticipant.USER:
            dialogue_acts = user_nlu.extract_dialogue_acts(utterance)
        else:
            dialogue_acts = agent_nlu.extract_dialogue_acts(utterance)
        utterance.add_dialogue_acts(dialogue_acts)
    return dialogue


def augment_dialogue(
    user_nlu: NLU,
    agent_nlu: NLU,
    information_need_annotator: InformationNeedAnnotator,
    dialogue: Dialogue,
) -> Dialogue:
    """Augments a dialogue with dialogue acts and information need.

    Args:
        user_nlu: NLU to use for user dialogue act annotation.
        agent_nlu: NLU to use for agent dialogue act annotation.
        information_need_annotator: Information need annotator.
        dialogue: Dialogue to augment.

    Returns:
        Augmented dialogue.
    """
    dialogue = annotate_dialogue_acts(user_nlu, agent_nlu, dialogue)
    dialogue = information_need_annotator.annotate_information_need(dialogue)
    return dialogue


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Augment the formatted dataset with additional information."
        ),
        prog="augment_dataset.py",
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the formatted dataset.",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to save the augmented dataset.",
    )
    parser.add_argument(
        "--user_nlu_config",
        type=str,
        default="config/nlu/user_dialogue_acts_extraction_config_default.yaml",
        help="NLU configuration file for user dialogue acts extraction.",
    )
    parser.add_argument(
        "--agent_nlu_config",
        type=str,
        default="config/nlu/agent_dialogue_acts_extraction_config_default.yaml",
        help="NLU configuration file for agent dialogue acts extraction.",
    )
    parser.add_argument(
        "--llm_interface_config",
        type=str,
        default="config/llm_interface/config_llm_interface_information_need.yaml",  # noqa: E501
        help="Configuration file for LLM interface for information need annotation.",  # noqa: E501
    )
    parser.add_argument(
        "--information_need_prompt",
        type=str,
        default=DEFAULT_INITIAL_PROMPT_MOVIES_FILE,
        help="File containing the prompt for information need annotation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dialogues = json_to_dialogues(args.input_path)

    user_nlu = NLU(get_dialogue_acts_extractor(args.user_nlu_config))
    agent_nlu = NLU(get_dialogue_acts_extractor(args.agent_nlu_config))

    with open(args.llm_interface_config, "r") as f:
        llm_interface_config = yaml.safe_load(f).get("llm_interface", {})

    information_need_llm_interface = get_llm_interface(llm_interface_config)
    information_need_annotator = InformationNeedAnnotator(
        information_need_llm_interface, args.information_need_prompt
    )

    augmented_dialogues = [
        augment_dialogue(
            user_nlu, agent_nlu, information_need_annotator, dialogue
        )
        for dialogue in tqdm(dialogues)
    ]

    with open(args.output_path, "w") as output_file:
        json.dump(
            [dialogue.to_dict() for dialogue in augmented_dialogues],
            output_file,
            indent=4,
        )
