"""LLM signal extractor for structured preference modeling."""

from __future__ import annotations

from typing import List, Optional

from usersimcrs.user_modeling.structured_preference.signal_extractor import (
    PreferenceSignal,
    PreferenceSignalsExtractor,
)


class LLMPreferenceSignalsExtractor(PreferenceSignalsExtractor):
    def extract(
        self,
        user_utterance: str,
        rating: float | None = None,
        past_dialogues: Optional[List[str]] = None,
    ) -> List[PreferenceSignal]:
        raise NotImplementedError
