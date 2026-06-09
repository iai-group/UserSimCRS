from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional, Tuple

from dialoguekit.participant.user_preferences import UserPreferences

from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.items.ratings import Ratings
from usersimcrs.user_modeling.preference_model import (
    KEY_ITEM_ID,
    PreferenceModel,
)


class StructuredPreferenceModel(PreferenceModel):
    """Preference model based on historical ratings and item metadata."""

    UPDATE_STEP = 0.25
    PREFERENCE_SCORE = 0.7
    NEW_PREFERENCE_MIN_CONFIRMATIONS = 2
    NEGATIVE_PATTERNS = (
        r"\b(?:avoid|nothing|not|no|without|not into|not too|too much)\b.*"
        r"\b{value}\b",
        r"\b{value}\s+(?:heavy|packed)\b",
    )
    POSITIVE_PATTERNS = (
        r"\b(?:like|love|prefer|enjoy)\b.*\b{value}\b",
        r"\b{value}\s+focused\b",
    )
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
        historical_ratings: Ratings,
        historical_user_id: Optional[str] = None,
        preference_threshold: float = PreferenceModel.PREFERENCE_THRESHOLD,
    ) -> None:
        """Initializes structured preferences.

        Args:
            domain: Domain.
            item_collection: Item collection.
            historical_ratings: Historical ratings.
            historical_user_id: Historical user ID. Defaults to None.
            preference_threshold: Minimum absolute score to store as
              meaningful preference.
        """
        super().__init__(
            domain, item_collection, historical_ratings, historical_user_id
        )
        self._preference_threshold = preference_threshold
        self._item_preferences = UserPreferences(self._user_id)
        self._slot_value_preferences = UserPreferences(self._user_id)
        self._slot_value_counts: Dict[Tuple[str, str], int] = {}
        self._pending_slot_value_updates: Dict[
            Tuple[str, str], List[float]
        ] = {}
        self._dialogue_buffer: List[str] = []
        self._initialize_preferences()

    @staticmethod
    def _normalize_preference_value(value: str) -> str:
        """Normalizes preference values for storage and matching.

        Args:
            value: Raw preference value.

        Returns:
            Value with normalized spacing and hyphens.
        """
        return " ".join(str(value).lower().replace("-", " ").split())

    def _initialize_preferences(self) -> None:
        """Initializes preferences from historical ratings.

        Builds the long-term item and slot-value profiles once at startup.
        """
        self._initialize_item_preferences()
        slot_value_ratings = self._collect_slot_value_ratings()
        self._initialize_slot_value_preferences(slot_value_ratings)

    def _initialize_item_preferences(self) -> None:
        """Stores strong historical item ratings as item preferences."""
        user_ratings = self._historical_ratings.get_user_ratings(
            self._historical_user_id
        )
        for item_id, rating in user_ratings.items():
            if abs(rating) < self._preference_threshold:
                continue
            self._item_preferences.set_preference(KEY_ITEM_ID, item_id, rating)

    def _collect_slot_value_ratings(self) -> Dict[str, Dict[str, List[float]]]:
        """Collects historical ratings for each slot-value pair.

        Returns:
            Nested mapping of slot -> value -> list of historical ratings.
        """
        slot_value_ratings: Dict[str, Dict[str, List[float]]] = {}
        user_ratings = self._historical_ratings.get_user_ratings(
            self._historical_user_id
        )
        for item_id, rating in user_ratings.items():
            item = self._item_collection.get_item(item_id)
            if item is None:
                continue

            for slot, value in self._iter_preference_values(item):
                slot_value_ratings.setdefault(slot, {}).setdefault(
                    value, []
                ).append(rating)
        return slot_value_ratings

    def _initialize_slot_value_preferences(
        self, slot_value_ratings: Dict[str, Dict[str, List[float]]]
    ) -> None:
        """Stores average slot-value ratings as slot-value preferences.

        Args:
            slot_value_ratings: Historical ratings grouped by slot and value.
        """
        for slot, value_ratings in slot_value_ratings.items():
            for value, ratings in value_ratings.items():
                score = sum(ratings) / len(ratings)
                self._slot_value_counts[(slot, value)] = len(ratings)
                if abs(score) >= self._preference_threshold:
                    self._slot_value_preferences.set_preference(
                        slot, value, score
                    )

    def _iter_preference_values(self, item) -> Iterable[Tuple[str, str]]:
        """Yields preference-relevant slot-value pairs for an item.

        Excludes title/name fields and normalizes multi-valued properties.
        """
        for slot in self._domain.get_slot_names():
            if slot.upper() in {"TITLE", "NAME"}:
                continue
            value = item.get_property(slot)
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            for slot_value in values:
                yield slot, self._normalize_preference_value(slot_value)

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

        Args:
            slot: Slot name.
            value: Slot value.

        Returns:
            Preference score for the slot-value pair, or 0 if unavailable.
        """
        self._assert_slot_exists(slot)
        value = self._normalize_preference_value(value)
        preference = self._slot_value_preferences.get_preference(slot, value)
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
        value = self._normalize_preference_value(value)

        key = (slot, value)
        old_score = self._slot_value_preferences.get_preference(slot, value)
        old_count = self._slot_value_counts.get(key, 0)

        if old_score is not None and old_count > 0:
            if score > old_score:
                new_score = min(old_score + self.UPDATE_STEP, score)
            else:
                new_score = max(old_score - self.UPDATE_STEP, score)
            self._slot_value_preferences.set_preference(slot, value, new_score)
            self._slot_value_counts[key] = old_count + 1
            return

        pending_scores = self._pending_slot_value_updates.setdefault(key, [])
        pending_scores.append(score)
        if len(pending_scores) < self.NEW_PREFERENCE_MIN_CONFIRMATIONS:
            return

        new_score = sum(pending_scores) / len(pending_scores)
        self._slot_value_preferences.set_preference(slot, value, new_score)
        self._slot_value_counts[key] = len(pending_scores)
        del self._pending_slot_value_updates[key]

    def _apply_text_update(self, text: str) -> None:
        """Applies preference updates for one buffered utterance.

        Args:
            text: User utterance collected during the dialogue.
        """
        text_lower = self._normalize_preference_value(text)
        for slot, value in self._extract_matched_values(text_lower):
            score = self._score_value_in_text(text_lower, value)
            if score != 0:
                self.update_slot_value_preference(slot, value, score)

    def _extract_matched_values(self, text_lower: str) -> List[Tuple[str, str]]:
        """Returns matched known preferences without shorter duplicate
        fragments.

        Args:
            text_lower: Lowercased user utterance.

        Returns:
            Matched slot-value pairs, preferring longer values over nested
            substrings.
        """
        matches: List[Tuple[str, str]] = []
        for slot, value, _ in self._rank_long_term_preferences():
            if len(value) >= 3 and value in text_lower:
                matches.append((slot, value))

        filtered_matches: List[Tuple[str, str]] = []
        for slot, value in sorted(matches, key=lambda match: -len(match[1])):
            if any(
                value != kept_value and value in kept_value
                for _, kept_value in filtered_matches
            ):
                continue
            filtered_matches.append((slot, value))
        return filtered_matches

    @staticmethod
    def _score_value_in_text(text_lower: str, value_lower: str) -> float:
        """Scores one matched value using nearby cues.

        Args:
            text_lower: Lowercased user utterance.
            value_lower: Lowercased slot value being inspected.

        Returns:
            A local preference score for the matched value.
        """
        contexts = [
            clause.strip()
            for clause in re.split(r"[.!?;]", text_lower)
            if value_lower in clause
        ]
        if not contexts:
            return 0

        for context in contexts:
            if any(
                re.search(pattern.format(value=re.escape(value_lower)), context)
                for pattern in StructuredPreferenceModel.NEGATIVE_PATTERNS
            ):
                return -StructuredPreferenceModel.PREFERENCE_SCORE
            if any(
                re.search(pattern.format(value=re.escape(value_lower)), context)
                for pattern in StructuredPreferenceModel.POSITIVE_PATTERNS
            ):
                return StructuredPreferenceModel.PREFERENCE_SCORE

        return 0

    def update_from_dialogue(self, dialogue) -> None:
        """Updates preferences after a completed dialogue.

        Buffers ordinary user turns and applies the accumulated updates once
        the dialogue ends with a stop utterance.

        Args:
            dialogue: Dialogue utterance object.
        """
        text = getattr(dialogue, "text", "").strip()
        if not text:
            return

        if text.lower() in self.DIALOGUE_STOP_TOKENS:
            for buffered_text in self._dialogue_buffer:
                self._apply_text_update(buffered_text)
            self._dialogue_buffer.clear()
            return

        self._dialogue_buffer.append(text)

    def _rank_long_term_preferences(self) -> List[Tuple[str, str, float]]:
        """Ranks long-term slot-value preferences by support and strength.

        Returns:
            Sorted list of long-term slot-value preferences.
        """
        ranked_preferences: List[Tuple[str, str, float]] = []
        for (
            slot,
            value_preferences,
        ) in self._slot_value_preferences._preferences.items():
            for value, score in value_preferences.items():
                ranked_preferences.append((slot, value, score))

        return sorted(
            ranked_preferences,
            key=lambda preference: (
                -math.log1p(
                    self._slot_value_counts.get(
                        (preference[0], preference[1]), 0
                    )
                ),
                -abs(preference[2]),
                preference[0],
                preference[1],
            ),
        )

    def get_preference_summary(self, max_preferences: int = 20) -> str:
        """Returns a preference summary for the LLM.

        Args:
            max_preferences: Total number of preferences.

        Returns:
            Summary for the LLM of long-term preferences.
        """
        ranked_preferences = self._rank_long_term_preferences()
        if not ranked_preferences:
            return ""

        max_long_term = max(1, max_preferences // 2)
        sections = [
            (
                "Positive preferences",
                [
                    preference
                    for preference in ranked_preferences
                    if preference[2] >= self._preference_threshold
                ][:max_long_term],
            ),
            (
                "Negative preferences",
                [
                    preference
                    for preference in ranked_preferences
                    if preference[2] <= -self._preference_threshold
                ][:max_long_term],
            ),
        ]
        summary_parts = [
            f"{title}: "
            + "; ".join(
                f"{slot}={value}:{score:.2f}"
                for slot, value, score in preferences
            )
            for title, preferences in sections
            if preferences
        ]
        return " | ".join(summary_parts)
