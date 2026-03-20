import argparse
from typing import Dict, List

import pandas as pd
from tqdm import tqdm
import yaml

from dialoguekit.participant.participant import DialogueParticipant
from dialoguekit.utils.dialogue_reader import json_to_dialogues
from scripts.nlu.metrics import (
    dialogue_acts_f1_score,
    dialogue_acts_precision,
    dialogue_acts_recall,
    intent_error_rate,
    slot_error_rate,
)
from usersimcrs.nlu.llm.llm_dialogue_act_extractor import (
    LLMDialogueActsExtractor,
)
from usersimcrs.utils.simulation_utils import get_llm_interface


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Experimentation with dialogue acts extraction."
    )

    parser.add_argument(
        "--user-extractor-config",
        type=str,
        default="config/nlu/user_dialogue_acts_extraction_config_default.yaml",
        help="User dialogue acts extractor configuration file.",
    )
    parser.add_argument(
        "--agent-extractor-config",
        type=str,
        default="config/nlu/agent_dialogue_acts_extraction_config_default.yaml",
        help="Agent dialogue acts extractor configuration file.",
    )
    parser.add_argument(
        "--annotated-dialogues",
        type=str,
        default="data/datasets/iard/formatted_IARD_annotated_gold.json",
        help="Annotated dialogues JSON file.",
    )
    return parser.parse_args()


def get_dialogue_acts_extractor(config_path: str) -> LLMDialogueActsExtractor:
    """Returns dialogue acts extractor based on given configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    interface_cfg = config.get("llm_interface", {})
    interface = get_llm_interface(interface_cfg)
    return LLMDialogueActsExtractor(interface, config_path)


if __name__ == "__main__":
    args = parse_args()

    dialogue_acts_extractors = {
        DialogueParticipant.USER: get_dialogue_acts_extractor(
            args.user_extractor_config
        ),
        DialogueParticipant.AGENT: get_dialogue_acts_extractor(
            args.agent_extractor_config
        ),
    }

    annotated_dialogues = json_to_dialogues(args.annotated_dialogues)

    print(f"Testing dialogue acts extraction on {args.annotated_dialogues}")

    metrics_list = ["ER_slot", "ER_intent", "Recall_DA", "Prec_DA", "F1_DA"]
    scores: Dict[str, Dict[str, List[float]]] = {
        participant: {m: [] for m in metrics_list}
        for participant in ["Global", "User", "Agent"]
    }

    for dialogue in tqdm(annotated_dialogues):
        for utterance in dialogue.utterances:
            participant = utterance.participant
            extractor = dialogue_acts_extractors.get(participant)

            if not extractor:
                raise RuntimeError(
                    f"Cannot access dialogue acts extractor for {participant}"
                )

            extracted_dialogue_acts = extractor.extract_dialogue_acts(utterance)

            metrics = {
                "ER_slot": slot_error_rate(
                    extracted_dialogue_acts, utterance.dialogue_acts
                ),
                "ER_intent": intent_error_rate(
                    extracted_dialogue_acts, utterance.dialogue_acts
                ),
                "Recall_DA": dialogue_acts_recall(
                    extracted_dialogue_acts, utterance.dialogue_acts
                ),
                "Prec_DA": dialogue_acts_precision(
                    extracted_dialogue_acts, utterance.dialogue_acts
                ),
                "F1_DA": dialogue_acts_f1_score(
                    extracted_dialogue_acts, utterance.dialogue_acts
                ),
            }

            target_keys = [
                "Global",
                "User" if participant == DialogueParticipant.USER else "Agent",
            ]
            for key in target_keys:
                for m_name, val in metrics.items():
                    scores[key][m_name].append(val)

    evaluation_results = pd.DataFrame(
        {k: {m: sum(v) / len(v) for m, v in scores[k].items()} for k in scores}
    )

    print("\nEvaluation results:")
    print(evaluation_results.round(3))
