"""Optimized LM for dialogue act extraction.

This module optimized a language model using DSpy for dialogue act extraction.
"""

import re
from typing import List, Union

import dspy

from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.intent import Intent
from dialoguekit.core.slot_value_annotation import SlotValueAnnotation


class DialogueActsSignature(dspy.Signature):
    __doc__ = (
        "Given an utterance, a list of possible intents and slots, extract a "
        "list of dialogue acts from it. A dialogue act is a pair of an intent "
        "and an optional list of slot-value pairs represented as "
        "'intent(slot=value,slot,...)' where the value of a slot is optional. "
        "Multiple dialogue acts are separated by '|'. If no dialogue acts are "
        "found, say '\n'."
    )

    input_utterance = dspy.InputField()
    intent_labels = dspy.InputField(
        prefix="Intent labels: ",
        desc="list of intent labels",
        format=lambda x: ", ".join(x) if isinstance(x, list) else x,
    )
    slot_labels = dspy.InputField(
        prefix="Slot labels: ",
        desc="list of slot labels",
        format=lambda x: ", ".join(x) if isinstance(x, list) else x,
    )

    dialogue_acts = dspy.OutputField(
        prefix="Dialogue acts: ",
        desc=(
            "list of dialogue acts formatted as: 'intent(slot=value,slot=value"
            ",...)|intent(slot=value,slot,...)|...'"
        ),
        format=lambda x: "|".join(x) if isinstance(x, list) else x,
    )


class ExtractionModule(dspy.Module):
    def __init__(
        self, intent_labels: List[str], slot_labels: List[str]
    ) -> None:
        """Initializes the extraction module.

        Args:
            intent_labels: List of intent labels.
            slot_labels: List of slot labels.
        """
        super(dspy.Module, self).__init__()

        self.intent_labels = intent_labels
        self.slot_labels = slot_labels

        self.cot = dspy.Predict(DialogueActsSignature)

    def validate_output_format(self, output: str) -> bool:
        """Validates the output format.

        The format is a list of dialogue acts formatted as:
        'intent(slot=value,slot=value,...)|intent(slot=value,slot,...)|...'

        Args:
            output: Output string.

        Returns:
            True if the output format is valid, False otherwise.
        """
        pattern = r"(\w+\((\w+(=\w+)?,?)*\))*\|?"
        return re.fullmatch(pattern, output) is not None

    def validate_intents(
        self, predicted_dialogue_acts: List[DialogueAct]
    ) -> bool:
        """Checks that all intents are valid.

        Args:
            predicted_dialogue_acts: Predicted dialogue acts.

        Returns:
            True if all intents are valid, False otherwise.
        """
        return all(
            dialogue_act.intent.label in self.intent_labels
            for dialogue_act in predicted_dialogue_acts
        )

    def validate_slot_value_pairs(
        self, utterance: str, predicted_dialogue_acts: List[DialogueAct]
    ) -> bool:
        """Checks that all slot-value pairs are valid.

        A slot-value pair is valid if the slot is in the slot labels and the
        value is in the utterance.

        Args:
            utterance: Input utterance.
            predicted_dialogue_acts: Predicted dialogue acts.

        Returns:
            True if all slot-value pairs are valid, False otherwise.
        """
        return all(
            annotation.slot in self.slot_labels
            and (annotation.value is None or annotation.value in utterance)
            for dialogue_act in predicted_dialogue_acts
            for annotation in dialogue_act.annotations
        )

    def forward(self, input_utterance: str) -> DialogueActsSignature:
        """Forward pass of the module.

        Args:
            input_utterance: Utterance text.


        Returns:
            Dialogue acts signature.
        """
        predicted_dialogue_acts = self.cot(
            input_utterance=input_utterance,
            intent_labels=self.intent_labels,
            slot_labels=self.slot_labels,
        )

        dspy.Suggest(
            self.validate_output_format(predicted_dialogue_acts.dialogue_acts),
            msg=(
                "The output format is invalid. Dialogue acts should be in the "
                "format: 'intent(slot=value,slot=value,...)|intent(slot=value,"
                "slot,...)|intent()|...'"
            ),
        )

        parsed_dialogue_acts = self.parse_dialogue_acts(
            predicted_dialogue_acts.dialogue_acts
        )

        # Use Suggest instead of Assert to avoid stopping the optimization
        # See: https://github.com/stanfordnlp/dspy/issues/1434
        dspy.Suggest(
            self.validate_intents(parsed_dialogue_acts),
            msg=(
                "Some dialogue acts have invalid intents. Intents should be in "
                f"{self.intent_labels}."
            ),
        )
        dspy.Suggest(
            self.validate_slot_value_pairs(
                input_utterance, parsed_dialogue_acts
            ),
            msg=(
                "Some dialogue acts have invalid slot-value pairs. A valid "
                f"slot-value pair should have a slot in {self.slot_labels} and "
                "no value, '?', or a value in the utterance."
            ),
        )

        return predicted_dialogue_acts

    def filter_invalid_dialogue_acts(
        self, utterance: str, dialogue_acts: List[DialogueAct]
    ) -> List[DialogueAct]:
        """Filters out invalid dialogue acts.

        This method should be used after inference to filter out dialogue
        acts with invalid intents, slots, or values.

        Args:
            utterance: Input utterance.
            dialogue_acts: List of dialogue acts.

        Returns:
            List of valid dialogue acts.
        """
        filtered_dialogue_acts = []
        for dialogue_act in dialogue_acts:
            if dialogue_act.intent.label not in self.intent_labels:
                continue
            filtered_annotations = []
            for annotation in dialogue_act.annotations:
                if annotation.slot not in self.slot_labels:
                    continue
                if (
                    annotation.value is not None
                    and annotation.value not in utterance
                ):
                    continue
                filtered_annotations.append(annotation)
            filtered_dialogue_acts.append(
                DialogueAct(dialogue_act.intent, filtered_annotations)
            )
        return filtered_dialogue_acts

    @classmethod
    def parse_dialogue_acts(
        cls, dialogue_acts: Union[str, List[str]]
    ) -> List[DialogueAct]:
        """Parses dialogue acts from a list of strings.

        Args:
            dialogue_acts: String representation of dialogue acts.

        Returns:
            List of dialogue acts.
        """
        parsed_dialogue_acts = []
        pattern = r"(\w+)\((.*)\)"
        dialogue_acts = (
            dialogue_acts.split("|")
            if not isinstance(dialogue_acts, list)
            else dialogue_acts
        )
        for dialogue_act in dialogue_acts:
            match = re.match(pattern, dialogue_act)
            if match:
                intent = Intent(match.group(1))
                slot_value_pairs = match.group(2).split(",")
                annotations = []
                for slot_value_pair in slot_value_pairs:
                    pair = slot_value_pair.split("=")
                    if len(pair) != 2:
                        # Invalid slot-value pair
                        continue

                    # Remove surrounding quotes if present
                    pair = [
                        re.sub(r'^"|"$', "", p) if isinstance(p, str) else p
                        for p in pair
                    ]
                    if pair[1] in ["", "None", "none", "?", "null"]:
                        pair[1] = None

                    annotations.append(SlotValueAnnotation(pair[0], pair[1]))
                parsed_dialogue_acts.append(
                    DialogueAct(intent=intent, annotations=annotations)
                )
        return parsed_dialogue_acts
