"""Interface to extract dialogue acts from an utterance."""

from abc import ABC, abstractmethod
from typing import List

from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.utterance import Utterance


class DialogueActsExtractor(ABC):
    def __init__(self) -> None:
        """Initializes the dialogue acts extractor."""
        super().__init__()

    @abstractmethod
    def extract_dialogue_acts(self, utterance: Utterance) -> List[DialogueAct]:
        """Extracts dialogue acts from an utterance.

        Args:
            utterance: Utterance.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Returns:
            List of dialogue acts.
        """
        raise NotImplementedError
