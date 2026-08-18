"""Signal extraction interfaces for structured preference modeling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from dialoguekit.core.utterance import Utterance


@dataclass
class PreferenceSignal:
    """Represents one extracted preference signal.

    Args:
        score: Signal score in [-1, 1].
        slot: Slot name for an explicit attribute signal.
        value: Slot value for an explicit attribute signal.
        item_id: Item identifier for an item-based signal.
        source: Signal source.
    """

    score: float
    slot: Optional[str] = None
    value: Optional[str] = None
    item_id: Optional[str] = None
    source: str = "attribute"


class PreferenceSignalsExtractor(ABC):
    @staticmethod
    def normalize_preference_value(value: str) -> str:
        """Normalizes a preference value for matching and memory keys.

        Args:
            value: Raw preference value.

        Returns:
            Normalized preference value.
        """
        return " ".join(str(value).lower().replace("-", " ").split())

    @abstractmethod
    def extract(self, utterance: Utterance) -> List[PreferenceSignal]:
        """Extracts preference signals from user input.

        Args:
            utterance: User utterance.

        Returns:
            Extracted preference signals.
        """
        raise NotImplementedError
