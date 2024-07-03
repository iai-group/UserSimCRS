"""Dialogue act extractor comprising intent classification and slot filling."""

from typing import List

from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.intent import Intent
from dialoguekit.core.slot_value_annotation import SlotValueAnnotation
from dialoguekit.core.utterance import Utterance
from dialoguekit.nlu.intent_classifier import IntentClassifier
from dialoguekit.nlu.slot_annotator import SlotValueAnnotator

from usersimcrs.nlu.dialogue_acts_extractor import DialogueActsExtractor


class SingleDialogueActExtractor(DialogueActsExtractor):
    def __init__(
        self,
        intent_classifier: IntentClassifier,
        slot_value_annotators: List[SlotValueAnnotator],
    ) -> None:
        """Initializes the dialogue act extractor.

        Args:
            intent_classifier: Intent classifier.
            slot_value_annotators: List of slot value pairs annotators.
        """
        super().__init__()
        self._intent_classifier = intent_classifier
        self._slot_value_annotators = slot_value_annotators

    def classify_intent(self, utterance: Utterance) -> Intent:
        """Classifies the intent of a given agent utterance."""
        return self._intent_classifier.classify_intent(utterance)

    def annotate_slot_values(
        self, utterance: Utterance
    ) -> List[SlotValueAnnotation]:
        """Annotates a given utterance with slot annotators.

        Args:
            utterance: Utterance to annotate.

        Returns:
            List of annotations.
        """
        annotation_list = []
        for slot_annotator in self._slot_value_annotators:
            annotation_list.extend(slot_annotator.get_annotations(utterance))
        return annotation_list

    def extract_dialogue_acts(self, utterance: Utterance) -> List[DialogueAct]:
        """Extracts one dialogue act from an utterance.

        Args:
            utterance: Utterance.

        Returns:
            List with one dialogue act.
        """
        intent = self.classify_intent(utterance)
        annotations = self.annotate_slot_values(utterance)
        return [DialogueAct(intent, annotations)]
