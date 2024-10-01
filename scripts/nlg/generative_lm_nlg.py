"""Script to experiment with LMGenerativeNLG."""

import argparse
from typing import List

import evaluate
from tqdm import tqdm

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.participant import DialogueParticipant
from dialoguekit.utils.dialogue_reader import json_to_dialogues
from usersimcrs.nlg.lm.nlg_generative_lm import LMGenerativeNLG


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Experimentation with generative NLG."
    )
    parser.add_argument(
        "--ollama-config-file",
        type=str,
        default="config/llm_interface/config_ollama_default.yaml",
        help="Ollama configuration file.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="data/datasets/iard/user_utterance_nlg_prompt.txt",
        help="Prompt file.",
    )
    parser.add_argument(
        "--prompt-prefix",
        type=str,
        default="Generated utterance:",
        help="Prefix to remove from generated utterances.",
    )
    parser.add_argument(
        "--input-dialogues",
        type=str,
        default="data/datasets/iard/formatted_IARD_annotated_gold.json",
        help="Input dialogues JSON file.",
    )
    return parser.parse_args()


def filter_user_utterances(dialogue: Dialogue) -> List[Utterance]:
    """Filters dialogue utterances to keep only user utterances.

    Args:
        dialogue: Dialogue.

    Returns:
        List of user utterances.
    """
    return [
        utterance
        for utterance in dialogue.utterances
        if utterance.participant == DialogueParticipant.USER
    ]


def compute_sacrebleu_score(
    gold_utterances: List[Utterance],
    generated_utterances: List[AnnotatedUtterance],
) -> float:
    """Computes the SacreBLEU score for the generated utterances.

    Args:
        gold_utterances: Gold utterances.
        generated_utterances: Generated utterances.

    Returns:
        SacreBLEU score.
    """
    metric = evaluate.load("sacrebleu")
    gold_nl_utterances = [[utterance.text] for utterance in gold_utterances]
    generated_nl_utterances = [
        utterance.text for utterance in generated_utterances
    ]
    return metric.compute(
        predictions=generated_nl_utterances, references=gold_nl_utterances
    )


def generate_utterances(
    nlg: LMGenerativeNLG, gold_utterances: List[Utterance]
) -> List[AnnotatedUtterance]:
    """Generates utterances using the NLG.

    Args:
        nlg: NLG.
        gold_utterances: Gold utterances.

    Returns:
        List of generated utterances.
    """
    generated_utterances = []
    for utterance in tqdm(gold_utterances):
        generated_utterances.append(
            nlg.generate_utterance_text(
                utterance.dialogue_acts,
                utterance.annotations,
            )
        )
    return generated_utterances


if __name__ == "__main__":
    args = parse_args()

    dialogues = json_to_dialogues(args.input_dialogues)
    user_utterances = []
    for dialogue in dialogues:
        user_utterances.extend(filter_user_utterances(dialogue))

    nlg = LMGenerativeNLG(
        args.ollama_config_file,
        args.prompt_file,
        args.prompt_prefix,
    )

    generated_utterances = generate_utterances(nlg, user_utterances)

    sacrebleu_score = compute_sacrebleu_score(
        user_utterances, generated_utterances
    )
    print(f"SacreBLEU score:\n{sacrebleu_score}")
