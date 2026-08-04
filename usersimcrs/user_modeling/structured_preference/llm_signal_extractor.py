"""LLM signal extractor for structured preference modeling."""

from __future__ import annotations

from typing import List

from usersimcrs.user_modeling.structured_preference.signal_extractor import (
    PreferenceSignal,
    PreferenceSignalsExtractor,
)


class LLMPreferenceSignalsExtractor(PreferenceSignalsExtractor):
    def extract(self, user_utterance: str) -> List[PreferenceSignal]:
        """Extracts preference signals from user text.

        Args:
            user_utterance: User utterance.

        Returns:
            Extracted preference signals.
        """
        raise NotImplementedError
