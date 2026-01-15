import argparse
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from dialoguekit.participant.participant import DialogueParticipant
from dialoguekit.utils.dialogue_reader import json_to_dialogues
from scripts.nlu.metrics import (
    dialogue_acts_f1_score,
    dialogue_acts_precision,
    dialogue_acts_recall,
    intent_error_rate,
    slot_error_rate,
)
# TODO: Update to new LLMDialogueActsExtractor class. Issue #230
from usersimcrs.nlu.lm.llm_dialogue_act_extractor import (
    LLMDialogueActsExtractor,
)


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
        default="data/datasets/iard/formatted_IARD_annotated_gold.json",
        help="Annotated dialogues JSON file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    user_dialogue_acts_extractor = LLMDialogueActsExtractor(
        args.user_extractor_config
    )
    agent_dialogue_acts_extractor = LLMDialogueActsExtractor(
        args.agent_extractor_config
    )

    annotated_dialogues = json_to_dialogues(args.annotated_dialogues)

    print(
        f"Testing dialogue acts extraction with {args.model} model on "
        f"{args.annotated_dialogues}"
    )

    scores: Dict[str, Dict[str, List[float]]] = dict.fromkeys(
        ["Global", "User", "Agent"],
        {
            "ER_slot": [],
            "ER_intent": [],
            "Recall_DA": [],
            "Prec_DA": [],
            "F1_DA": [],
        },
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

            scores["Global"]["ER_slot"].append(slot_error_rate_score)
            scores["Global"]["ER_intent"].append(intent_error_rate_score)
            scores["Global"]["Recall_DA"].append(dialogue_acts_recall_score)
            scores["Global"]["Prec_DA"].append(dialogue_acts_precision_score)
            scores["Global"]["F1_DA"].append(dialogue_acts_f1_score_score)

            if utterance.participant == DialogueParticipant.USER:
                scores["User"]["ER_slot"].append(slot_error_rate_score)
                scores["User"]["ER_intent"].append(intent_error_rate_score)
                scores["User"]["Recall_DA"].append(dialogue_acts_recall_score)
                scores["User"]["Prec_DA"].append(dialogue_acts_precision_score)
                scores["User"]["F1_DA"].append(dialogue_acts_f1_score_score)
            else:
                scores["Agent"]["ER_slot"].append(slot_error_rate_score)
                scores["Agent"]["ER_intent"].append(intent_error_rate_score)
                scores["Agent"]["Recall_DA"].append(dialogue_acts_recall_score)
                scores["Agent"]["Prec_DA"].append(dialogue_acts_precision_score)
                scores["Agent"]["F1_DA"].append(dialogue_acts_f1_score_score)

    evaluation_results = pd.DataFrame(
        {k: {m: sum(v) / len(v) for m, v in scores[k].items()} for k in scores}
    )

    print("\nEvaluation results:")
    print(evaluation_results.round(3))
