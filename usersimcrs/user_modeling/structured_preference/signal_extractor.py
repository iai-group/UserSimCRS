"""Signal extraction interfaces for structured preference modeling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


def normalize_preference_value(value: str) -> str:
    return " ".join(str(value).lower().replace("-", " ").split())


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
    @abstractmethod
    def extract(
        self,
        user_utterance: str,
        rating: float | None = None,
        past_dialogues: Optional[List[str]] = None,
    ) -> List[PreferenceSignal]:
        """Extracts preference signals from user input.

        Args:
            user_utterance: User utterance.
            rating: Optional rating signal.
            past_dialogues: Optional past dialogue texts.

        Returns:
            Extracted preference signals.
        """
        raise NotImplementedError
