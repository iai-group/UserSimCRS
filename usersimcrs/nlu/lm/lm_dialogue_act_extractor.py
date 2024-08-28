"""Dialogue act extractor based on a language model."""

import functools
from copy import deepcopy
from typing import List

import dspy
from dspy.primitives.assertions import (
    assert_transform_module,
    backtrack_handler,
)

from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.utterance import Utterance
from dialoguekit.nlu.dialogue_acts_extractor import DialogueActsExtractor
from usersimcrs.nlu.lm.lm_nlu import ExtractionModule


class LMDialogueActsExtractor(DialogueActsExtractor):
    def __init__(
        self,
        model_name: str,
        ollama_base_url: str,
        lm_nlu_module: ExtractionModule,
    ) -> None:
        """Initializes the dialogue act extractor.

        Args:
            model_name: Name of the language model.
            ollama_base_url: Base URL of the Ollama service.
            lm_nlu_module: Language model NLU module.
        """
        super().__init__()

        self.lm = dspy.OllamaLocal(model=model_name, base_url=ollama_base_url)
        self.lm_nlu_module = deepcopy(lm_nlu_module)
        self.lm_nlu_module = assert_transform_module(
            self.lm_nlu_module,
            functools.partial(backtrack_handler, max_backtracks=3),
        )

    def extract_dialogue_acts(self, utterance: Utterance) -> List[DialogueAct]:
        """Extracts dialogue acts from an utterance.

        Args:
            utterance: Utterance.

        Returns:
            List of dialogue acts.
        """
        with dspy.context(lm=self.lm):
            prediction = self.lm_nlu_module(utterance.text)
            predicted_dialogue_acts = ExtractionModule.parse_dialogue_acts(
                prediction.dialogue_acts
            )
            predicted_dialogue_acts = (
                self.lm_nlu_module.filter_invalid_dialogue_acts(
                    utterance.text,
                    predicted_dialogue_acts,
                )
            )
        return predicted_dialogue_acts

    def save(self, path: str) -> None:
        """Saves the dialogue act extractor to a given path.

        Not implemented as the language model is externally hosted and the NLU
        module has its own saving mechanism.

        Args:
            path: Path to save the dialogue act extractor.
        """
        pass

    @classmethod
    def load(cls, path: str) -> DialogueActsExtractor:
        """Loads the dialogue act extractor from a path.

        Not implemented as the language model is externally hosted and the NLU
        module has its own loading mechanism.

        Args:
            path: Path to the dialogue act extractor.
        """
        pass
