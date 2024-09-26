import argparse
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.intent import Intent
from dialoguekit.core.slot_value_annotation import SlotValueAnnotation
from dialoguekit.participant.participant import DialogueParticipant
from dialoguekit.utils.dialogue_reader import json_to_dialogues
from scripts.nlu.metrics import (
    dialogue_acts_f1_score,
    dialogue_acts_precision,
    dialogue_acts_recall,
    intent_error_rate,
    slot_error_rate,
)
from usersimcrs.nlu.lm.lm_dialogue_act_extractor import LMDialogueActsExtractor


def parse_gold_dialogue_acts(
    json_dialogue_acts: List[Dict[str, Any]]
) -> List[DialogueAct]:
    """Parses gold dialogue acts from json format.

    Args:
        json_dialogue_acts: List of dialogue acts in JSON format.

    Returns:
        List of dialogue acts.
    """
    dialogue_acts = []
    for json_dialogue_act in json_dialogue_acts:
        intent = Intent(json_dialogue_act["intent"])
        annotations = []
        slot_values = json_dialogue_act.get("slot_values", [])
        if slot_values:
            for json_annotation in slot_values:
                annotations.append(
                    SlotValueAnnotation(
                        json_annotation[0],
                        json_annotation[1],
                        json_annotation[2],
                        json_annotation[3],
                    )
                )
        dialogue_acts.append(DialogueAct(intent, annotations))
    return dialogue_acts


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Experimentation with dialogue acts extraction."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="mistral-nemo:latest",
        help="Model to evaluate.",
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
        default="data/iard/formatted_IARD_annotated_gold.json",
        help="Annotated dialogues JSON file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    user_dialogue_acts_extractor = LMDialogueActsExtractor(
        args.user_extractor_config
    )
    agent_dialogue_acts_extractor = LMDialogueActsExtractor(
        args.agent_extractor_config
    )

    annotated_dialogues = json_to_dialogues(args.annotated_dialogues)

    print(
        f"Testing dialogue acts extraction with {args.model} model on "
        f"{args.annotated_dialogues}"
    )

    scores = dict.fromkeys(
        ["Global", "User", "Agent"],
        {"SER": [], "IER": [], "DAR": [], "DAP": [], "DAF1": []},
    )

    for dialogue in tqdm(annotated_dialogues):
        for utterance in dialogue.utterances:
            if utterance.participant == DialogueParticipant.USER:
                extracted_dialogue_acts = (
                    user_dialogue_acts_extractor.extract_dialogue_acts(
                        utterance
                    )
                )
            else:
                extracted_dialogue_acts = (
                    agent_dialogue_acts_extractor.extract_dialogue_acts(
                        utterance
                    )
                )

            slot_error_rate_score = slot_error_rate(
                extracted_dialogue_acts, utterance.dialogue_acts
            )
            intent_error_rate_score = intent_error_rate(
                extracted_dialogue_acts, utterance.dialogue_acts
            )
            dialogue_acts_recall_score = dialogue_acts_recall(
                extracted_dialogue_acts, utterance.dialogue_acts
            )
            dialogue_acts_precision_score = dialogue_acts_precision(
                extracted_dialogue_acts, utterance.dialogue_acts
            )
            dialogue_acts_f1_score_score = dialogue_acts_f1_score(
                extracted_dialogue_acts, utterance.dialogue_acts
            )

            scores["Global"]["SER"].append(slot_error_rate_score)
            scores["Global"]["IER"].append(intent_error_rate_score)
            scores["Global"]["DAR"].append(dialogue_acts_recall_score)
            scores["Global"]["DAP"].append(dialogue_acts_precision_score)
            scores["Global"]["DAF1"].append(dialogue_acts_f1_score_score)

            if utterance.participant == DialogueParticipant.USER:
                scores["User"]["SER"].append(slot_error_rate_score)
                scores["User"]["IER"].append(intent_error_rate_score)
                scores["User"]["DAR"].append(dialogue_acts_recall_score)
                scores["User"]["DAP"].append(dialogue_acts_precision_score)
                scores["User"]["DAF1"].append(dialogue_acts_f1_score_score)
            else:
                scores["Agent"]["SER"].append(slot_error_rate_score)
                scores["Agent"]["IER"].append(intent_error_rate_score)
                scores["Agent"]["DAR"].append(dialogue_acts_recall_score)
                scores["Agent"]["DAP"].append(dialogue_acts_precision_score)
                scores["Agent"]["DAF1"].append(dialogue_acts_f1_score_score)

    evaluation_results = pd.DataFrame(
        {k: {m: sum(v) / len(v) for m, v in scores[k].items()} for k in scores}
    )

    print("\nEvaluation results:")
    print(evaluation_results.round(3))
