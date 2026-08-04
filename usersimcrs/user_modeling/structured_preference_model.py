"""Structured preference model backed by historical ratings and metadata."""

from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.user_preferences import UserPreferences

from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item import Item
from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.items.ratings import Ratings
from usersimcrs.user_modeling.preference_model import (
    KEY_ITEM_ID,
    PreferenceModel,
)
from usersimcrs.user_modeling.structured_preference import (
    HeuristicPreferenceSignalsExtractor,
    PreferenceMemory,
    PreferenceSignal,
    PreferenceSignalsExtractor,
)


class StructuredPreferenceModel(PreferenceModel):

    UPDATE_STEP = 0.25
    LONG_TERM_PROMOTION_MIN_CONFIRMATIONS = 2
    RATING_HISTORY_WEIGHT = 1.0
    DIALOGUE_HISTORY_WEIGHT = 1.0
    DIALOGUE_STOP_TOKENS = {
        "exit",
        "goodbye",
        "bye",
        "quit",
        "stop",
        "end",
        "giveup",
    }

    def __init__(
        self,
        domain: SimulationDomain,
        item_collection: ItemCollection,
        historical_ratings: Optional[Ratings] = None,
        historical_user_id: Optional[str] = None,
        preference_threshold: float = PreferenceModel.PREFERENCE_THRESHOLD,
        dialogue_history: Optional[Iterable[Any]] = None,
        signals_extractor: Optional[PreferenceSignalsExtractor] = None,
        rating_history_weight: float = RATING_HISTORY_WEIGHT,
        dialogue_history_weight: float = DIALOGUE_HISTORY_WEIGHT,
    ) -> None:
        """Initializes structured preferences.

        Args:
            domain: Domain.
            item_collection: Item collection.
            historical_ratings: Optional historical ratings.
            historical_user_id: Historical user ID. Defaults to None.
            preference_threshold: Minimum absolute score to store as a
              meaningful preference. Defaults to
              `PreferenceModel.PREFERENCE_THRESHOLD`.
            dialogue_history: Optional previous dialogues or user utterances to
              initialize session preferences from. Defaults to None.
            signals_extractor: Optional preference signal extractor. Defaults
              to a heuristic extractor.
            rating_history_weight: Weight for evidence extracted from
              historical ratings. Defaults to 1.0.
            dialogue_history_weight: Weight for evidence extracted from
              historical dialogues. Defaults to 1.0.
        """
        historical_ratings = historical_ratings or Ratings(item_collection)
        if historical_user_id is None and not historical_ratings._user_ratings:
            historical_user_id = "dialogue_history"
        super().__init__(
            domain, item_collection, historical_ratings, historical_user_id
        )
        self._preference_threshold = preference_threshold
        self._dialogue_history = list(dialogue_history or [])
        self._rating_history_weight = rating_history_weight
        self._dialogue_history_weight = dialogue_history_weight
        self._item_preferences = UserPreferences(self._user_id)
        self._long_term_preferences = PreferenceMemory(self._user_id)
        self._session_preferences = PreferenceMemory(self._user_id)
        self._signals_extractor = (
            signals_extractor
            or HeuristicPreferenceSignalsExtractor(domain, item_collection)
        )
        self._initialize_preferences()

    def _initialize_preferences(self) -> None:
        """Initializes preferences from historical ratings and dialogues."""
        self._long_term_preferences.set_weighted_scores(
            self._rating_scores(),
            self._dialogue_scores(),
            self._rating_history_weight,
            self._dialogue_history_weight,
            self._preference_threshold,
        )

    def _rating_scores(self) -> Dict[Tuple[str, str], List[float]]:
        """Returns preference scores extracted from historical ratings.

        Returns:
            Rating scores keyed by slot-value pair.
        """
        rating_scores: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for item_id, rating in self._historical_ratings.get_user_ratings(
            self._historical_user_id
        ).items():
            item = self._item_collection.get_item(item_id)
            if item is None:
                continue
            if abs(rating) >= self._preference_threshold:
                self._item_preferences.set_preference(
                    KEY_ITEM_ID, item_id, rating
                )
            for slot, value in self._item_slot_values(item):
                rating_scores[(slot, value)].append(rating)
        return rating_scores

    def _dialogue_scores(self) -> Dict[Tuple[str, str], List[float]]:
        """Returns preference scores extracted from historical dialogues.

        Returns:
            Dialogue scores keyed by slot-value pair.
        """
        dialogue_scores: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for text in self._iter_dialogue_history_texts():
            for signal in self._extract_slot_signals_from_text(text):
                if not signal.slot or signal.value is None:
                    continue
                dialogue_scores[(signal.slot, signal.value)].append(
                    signal.score
                )
        return dialogue_scores

    def _extract_slot_signals_from_text(
        self, text: str
    ) -> List[PreferenceSignal]:
        """Extracts slot-value preference signals from one text.

        Args:
            text: User utterance text.

        Returns:
            Slot-value preference signals.
        """
        text = text.strip()
        if not text or self._is_stop_token(text):
            return []

        signals = self._signals_extractor.extract(text)
        return list(self._slot_preference_signals(signals))

    def _is_stop_token(self, text: str) -> bool:
        """Checks whether text ends preference updates.

        Args:
            text: User utterance text.

        Returns:
            True if text is a stop token, otherwise False.
        """
        normalized_text = self._signals_extractor.normalize_preference_value(
            text
        )
        return normalized_text in self.DIALOGUE_STOP_TOKENS

    def _item_slot_values(self, item: Item) -> List[Tuple[str, str]]:
        """Returns normalized preference slot values for an item.

        Args:
            item: Item to read.

        Returns:
            Slot-value pairs from informable item properties.
        """
        slot_values = []
        for slot in self._domain.get_informable_slots():
            value = item.get_property(slot)
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            for entry in values:
                normalized_value = (
                    self._signals_extractor.normalize_preference_value(entry)
                )
                slot_values.append((slot, normalized_value))
        return slot_values

    def _iter_dialogue_history_texts(self) -> List[str]:
        """Returns user utterance texts from supported history formats.

        Returns:
            User utterance texts.
        """
        texts = []
        for entry in self._dialogue_history:
            if isinstance(entry, str):
                texts.append(entry)
                continue

            utterances = getattr(entry, "utterances", None)
            if utterances is not None:
                for utterance in utterances:
                    participant = getattr(utterance, "participant", None)
                    participant_name = getattr(participant, "name", None)
                    if participant_name and participant_name != "USER":
                        continue
                    texts.append(getattr(utterance, "text", ""))
                continue

            if isinstance(entry, dict):
                if "conversation" in entry:
                    for utterance in entry["conversation"]:
                        if utterance.get("participant") != "USER":
                            continue
                        texts.append(utterance.get("utterance", ""))
                    continue
                texts.append(entry.get("utterance", entry.get("text", "")))
                continue

            texts.append(getattr(entry, "text", ""))
        return texts

    def _slot_preference_signals(
        self, signals: Iterable[PreferenceSignal]
    ) -> List[PreferenceSignal]:
        """Converts extracted signals to slot-value preference signals.

        Args:
            signals: Extracted preference signals.

        Returns:
            Slot-value preference signals.
        """
        signals = list(signals)
        explicit_keys = set()
        slot_signals = []
        for signal in signals:
            if not signal.slot or signal.value is None:
                continue
            normalized_value = (
                self._signals_extractor.normalize_preference_value(signal.value)
            )
            explicit_keys.add((signal.slot, normalized_value))
            slot_signals.append(
                PreferenceSignal(
                    slot=signal.slot,
                    value=normalized_value,
                    score=max(-1.0, min(1.0, signal.score)),
                    source=signal.source,
                )
            )

        for signal in signals:
            if not signal.item_id:
                continue
            item = self._item_collection.get_item(signal.item_id)
            if item is None:
                continue
            for slot, value in self._item_slot_values(item):
                if (slot, value) in explicit_keys:
                    continue
                slot_signals.append(
                    PreferenceSignal(
                        slot=slot,
                        value=value,
                        score=max(-1.0, min(1.0, signal.score)),
                        source=signal.source,
                    )
                )
        return slot_signals

    def _promote_session_preferences(self) -> None:
        """Promotes confirmed session preferences to long-term memory."""
        session_items = list(self._session_preferences.items())
        for slot, value, score, count in session_items:
            if count < self.LONG_TERM_PROMOTION_MIN_CONFIRMATIONS:
                continue

            long_term_score = self._long_term_preferences.get(slot, value)
            if long_term_score is None:
                self._long_term_preferences.set(slot, value, score, count)
                continue

            long_term_count = self._long_term_preferences.get_count(slot, value)
            total_count = long_term_count + count
            merged_score = (
                long_term_score * long_term_count + score * count
            ) / total_count
            self._long_term_preferences.set(
                slot, value, merged_score, total_count
            )

        self._session_preferences.clear()

    def get_item_preference(self, item_id: str) -> float:
        """Returns preference score for an item.

        Args:
            item_id: Item identifier.

        Returns:
            Long-term item preference if available, otherwise 0.
        """
        self._assert_item_exists(item_id)
        preference = self._item_preferences.get_preference(KEY_ITEM_ID, item_id)
        return preference if preference is not None else 0

    def get_slot_value_preference(self, slot: str, value: str) -> float:
        """Returns preference score for a slot-value pair.

        Session preferences are checked first because they capture the current
        dialogue context; long-term preferences are used as a fallback.

        Args:
            slot: Slot name.
            value: Slot value.

        Returns:
            Preference score for the slot-value pair, or 0 if unavailable.
        """
        self._assert_slot_exists(slot)
        normalized_value = self._signals_extractor.normalize_preference_value(
            value
        )
        preference = self._session_preferences.get(slot, normalized_value)
        if preference is None:
            preference = self._long_term_preferences.get(slot, normalized_value)
        return preference if preference is not None else 0

    def update_slot_value_preference(
        self, slot: str, value: str, score: float
    ) -> None:
        """Updates one slot-value preference.

        Args:
            slot: Slot name.
            value: Slot value.
            score: Preference score to store.
        """
        self._assert_slot_exists(slot)
        self._session_preferences.increment_towards(
            slot,
            self._signals_extractor.normalize_preference_value(value),
            max(-1.0, min(1.0, score)),
            self.UPDATE_STEP,
        )

    def get_preference_summary(self, max_preferences: int = 20) -> str:
        """Returns a preference summary for the LLM.

        Args:
            max_preferences: Total number of preferences. Defaults to 20.

        Returns:
            Summary for the LLM of long-term preferences.
        """
        session_preferences = self._session_preferences.ranked_items()[
            : max(1, max_preferences // 2)
        ]
        long_term_preferences = self._long_term_preferences.ranked_items()
        positive_preferences = [
            preference
            for preference in long_term_preferences
            if preference[2] >= self._preference_threshold
        ][: max(1, max_preferences // 2)]
        negative_preferences = [
            preference
            for preference in long_term_preferences
            if preference[2] <= -self._preference_threshold
        ][: max(1, max_preferences // 2)]

        sections = [
            ("Session preferences", session_preferences),
            ("Positive preferences", positive_preferences),
            ("Negative preferences", negative_preferences),
        ]
        return " | ".join(
            f"{title}: "
            + "; ".join(
                f"{slot}={value}:{score:.2f}"
                for slot, value, score in preferences
            )
            for title, preferences in sections
            if preferences
        )

    def update_from_dialogue(self, utterance: Utterance) -> None:
        """Updates preferences from one user turn.

        Args:
            utterance: User utterance used to update preferences.
        """
        text = getattr(utterance, "text", "")
        if self._is_stop_token(text):
            self._promote_session_preferences()
            return

        for signal in self._extract_slot_signals_from_text(text):
            if not signal.slot or signal.value is None:
                continue
            self.update_slot_value_preference(
                signal.slot, signal.value, signal.score
            )
