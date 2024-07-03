"""Interface to annotate an utterance.

The annotations do not include dialogue acts that have their own extractor."""

from abc import ABC, abstractmethod
from typing import List

from dialoguekit.core.annotation import Annotation
from dialoguekit.core.utterance import Utterance


class Annotator(ABC):
    def __init__(self) -> None:
        """Initializes the annotator."""
        super().__init__()

    @abstractmethod
    def annotate(self, utterance: Utterance) -> List[Annotation]:
        """Annotates an utterance.

        Args:
            utterance: Utterance.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Returns:
            List of annotations.
        """
        raise NotImplementedError
