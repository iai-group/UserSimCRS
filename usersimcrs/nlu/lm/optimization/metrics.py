"""Define the evaluation metric for the DSpy NLU module.

We define 5 metrics:
1. Slot Error Rate (SER)
2. Intent Error Rate (IER)
3. Dialogue Acts Recall (DAR)
4. Dialogue Acts Precision (DAP)
5. Dialogue Acts F1 Score (DAF1)
"""

from typing import List, Tuple

import dspy

from dialoguekit.core.dialogue_act import DialogueAct
from usersimcrs.nlu.lm.lm_nlu import DialogueActsSignature, ExtractionModule


def _get_slot_value_pairs(
    dialogue_acts: List[DialogueAct],
) -> List[Tuple[str, str]]:
    """Gets the slot-value pairs from given dialogue acts."""
    return [
        (da.intent.label, annotation.slot, annotation.value)
        for da in dialogue_acts
        for annotation in da.annotations
    ]


def _get_intents(dialogue_acts: List[DialogueAct]) -> List[str]:
    """Gets the intents from given dialogue acts."""
    return [da.intent.label for da in dialogue_acts]


def slot_error_rate(
    predicted_dialogue_acts: List[DialogueAct],
    target_dialogue_acts: List[DialogueAct],
) -> float:
    """Calculates the Slot Error Rate (SER).

    Args:
        predicted_dialogue_acts: Predicted dialogue acts.
        target_dialogue_acts: Target dialogue acts.

    Returns:
        Slot Error Rate (SER).
    """
    predicted_slot_value_pairs = _get_slot_value_pairs(predicted_dialogue_acts)
    target_slot_value_pairs = _get_slot_value_pairs(target_dialogue_acts)

    num_correct_slot_value_pairs = len(
        set(predicted_slot_value_pairs) & set(target_slot_value_pairs)
    )
    num_total_slot_value_pairs = len(target_slot_value_pairs)
    if num_total_slot_value_pairs == 0:
        return 0.0
    return 1 - num_correct_slot_value_pairs / num_total_slot_value_pairs


def intent_error_rate(
    predicted_dialogue_acts: List[DialogueAct],
    target_dialogue_acts: List[DialogueAct],
) -> float:
    """Calculates the Intent Error Rate (IER).

    Args:
        predicted_dialogue_acts: Predicted dialogue acts.
        target_dialogue_acts: Target dialogue acts.

    Returns:
        Intent Error Rate (IER).
    """
    predicted_intents = _get_intents(predicted_dialogue_acts)
    target_intents = _get_intents(target_dialogue_acts)

    num_correct_intents = len(set(predicted_intents) & set(target_intents))
    num_total_intents = len(target_intents)
    if num_total_intents == 0:
        return 0.0
    return 1 - num_correct_intents / num_total_intents


def dialogue_acts_recall(
    predicted_dialogue_acts: List[DialogueAct],
    target_dialogue_acts: List[DialogueAct],
) -> float:
    """Calculates the Dialogue Acts Recall (DAR).

    Args:
        predicted_dialogue_acts: Predicted dialogue acts.
        target_dialogue_acts: Target dialogue acts.

    Returns:
        Dialogue Acts Recall (DAR).
    """
    num_correct_dialogue_acts = len(
        set(predicted_dialogue_acts) & set(target_dialogue_acts)
    )
    num_total_dialogue_acts = len(target_dialogue_acts)
    if num_total_dialogue_acts == 0:
        return 0.0
    return num_correct_dialogue_acts / num_total_dialogue_acts


def dialogue_acts_precision(
    predicted_dialogue_acts: List[DialogueAct],
    target_dialogue_acts: List[DialogueAct],
) -> float:
    """Calculates the Dialogue Acts Precision (DAP).

    Args:
        predicted_dialogue_acts: Predicted dialogue acts.
        target_dialogue_acts: Target dialogue acts.

    Returns:
        Dialogue Acts Precision (DAP).
    """
    num_correct_dialogue_acts = len(
        set(predicted_dialogue_acts) & set(target_dialogue_acts)
    )
    num_total_dialogue_acts = len(predicted_dialogue_acts)
    if num_total_dialogue_acts == 0:
        return 0.0
    return num_correct_dialogue_acts / num_total_dialogue_acts


def dialogue_acts_f1_score(
    predicted_dialogue_acts: List[DialogueAct],
    target_dialogue_acts: List[DialogueAct],
) -> float:
    """Calculates the Dialogue Acts F1 Score (DAF1).

    Args:
        predicted_dialogue_acts: Predicted dialogue acts.
        target_dialogue_acts: Target dialogue acts.

    Returns:
        Dialogue Acts F1 Score (DAF1).
    """
    recall = dialogue_acts_recall(
        predicted_dialogue_acts, target_dialogue_acts
    )
    precision = dialogue_acts_precision(
        predicted_dialogue_acts, target_dialogue_acts
    )
    if precision + recall == 0.0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def validate_answer(
    example: dspy.Example,
    prediction: DialogueActsSignature,
    trace: object = None,
) -> float:
    """Calculates the Dialogue Acts F1 Score (DAF1).

    Args:
        example: Example with gold dialogue acts.
        prediction: Module's prediction.
        trace: Unused, kept for compatibility.

    Returns:
        Dialogue Acts F1 Score (DAF1).
    """
    target_dialogue_acts = ExtractionModule.parse_dialogue_acts(
        example.dialogue_acts
    )

    predicted_dialogue_acts = ExtractionModule.parse_dialogue_acts(
        prediction.dialogue_acts
    )

    return dialogue_acts_f1_score(
        predicted_dialogue_acts, target_dialogue_acts
    )
