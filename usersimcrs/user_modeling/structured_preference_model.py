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
    LONG_TERM_PROMOTION_MIN_CONFIRMATIONS = 2
    NEGATIVE_PATTERNS = (
        r"\b(?:avoid|nothing|not|no|without|not into|not too|too much)\b.*"
        r"\b{value}\b",
        r"\bnot interested in\b.*\b{value}\b",
        r"\b(?:anything but|other than|except|rather than)\b.*\b{value}\b",
        r"\bnot a[n]?\b.*\b{value}\b",
        r"\bnot in\b.*\b{value}\b(?:\s+genre)?\b",
        r"\bwithout\b.*\b{value}\b",
        r"\b{value}\s+(?:heavy|packed)\b",
    )
    POSITIVE_PATTERNS = (
        r"\b(?:like|love|prefer|enjoy)\b.*\b{value}\b",
        r"\blooking for\b.*\b{value}\b",
        r"\binterested in\b.*\b{value}\b",
        r"\breally looking for\b.*\b{value}\b",
        r"\bcan you recommend\b.*\b{value}\b",
        r"\bcould you suggest\b.*\b{value}\b",
        r"\b(?:with|about|featuring|set in|centered around|based on)\b.*"
        r"\b{value}\b",
        r"\bthemes? of\b.*\b{value}\b",
        r"\bmore\b.*\b{value}\b",
        r"\bwithin the\b.*\b{value}\b(?:\s+genre)?\b",
        r"\bin the\b.*\b{value}\b(?:\s+genre)?\b",
        r"\b{value}\b.*\bover other genres\b",
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
        self._long_term_slot_value_preferences = UserPreferences(self._user_id)
        self._long_term_slot_value_counts: Dict[Tuple[str, str], int] = {}
        self._session_slot_value_preferences = UserPreferences(self._user_id)
        self._session_slot_value_counts: Dict[Tuple[str, str], int] = {}
        self._catalog_slot_values = self._collect_catalog_slot_values()
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

    def _collect_catalog_slot_values(self) -> Dict[str, List[str]]:
        """Collects normalized slot-values available in the item catalog."""
        catalog_slot_values: Dict[str, List[str]] = {}
        for slot in self._domain.get_slot_names():
            if slot.upper() in {"TITLE", "NAME"}:
                continue
            normalized_values = {
                normalized_value
                for value in self._item_collection.get_possible_property_values(
                    slot
                )
                if value is not None
                for normalized_value in [
                    self._normalize_preference_value(value)
                ]
                if len(normalized_value) >= 3
            }
            catalog_slot_values[slot] = sorted(
                normalized_values,
                key=len,
                reverse=True,
            )
        return catalog_slot_values

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
                self._long_term_slot_value_counts[(slot, value)] = len(ratings)
                if abs(score) >= self._preference_threshold:
                    self._long_term_slot_value_preferences.set_preference(
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
        preference = self._session_slot_value_preferences.get_preference(
            slot, value
        )
        if preference is None:
            preference = self._long_term_slot_value_preferences.get_preference(
                slot, value
            )
        return preference if preference is not None else 0

    def update_slot_value_preference(
        self, slot: str, value: str, score: float
    ) -> None:
        """Updates one slot-value preference.

        Existing long-term preferences are updated in place. New preferences
        are first tracked in the session layer and may later be promoted to
        long-term memory.

        Args:
            slot: Slot name.
            value: Slot value.
            score: Preference score to store.
        """
        self._assert_slot_exists(slot)
        value = self._normalize_preference_value(value)

        if (
            self._long_term_slot_value_preferences.get_preference(slot, value)
            is not None
        ):
            self._update_existing_preference(
                self._long_term_slot_value_preferences,
                self._long_term_slot_value_counts,
                slot,
                value,
                score,
            )
            return

        self._update_session_preference(slot, value, score)

    def _update_existing_preference(
        self,
        preference_store: UserPreferences,
        preference_counts: Dict[Tuple[str, str], int],
        slot: str,
        value: str,
        score: float,
    ) -> None:
        """Updates an already stored preference by moving it toward a signal."""
        key = (slot, value)
        old_score = preference_store.get_preference(slot, value)
        old_count = preference_counts.get(key, 0)

        if old_score is None:
            preference_store.set_preference(slot, value, score)
            preference_counts[key] = 1
            return

        direction = 1 if score > 0 else -1
        new_score = max(
            -1.0, min(1.0, old_score + direction * self.UPDATE_STEP)
        )
        preference_store.set_preference(slot, value, new_score)
        preference_counts[key] = max(1, old_count) + 1

    def _update_session_preference(
        self, slot: str, value: str, score: float
    ) -> None:
        """Tracks a new preference inside the current dialogue session."""
        key = (slot, value)
        old_score = self._session_slot_value_preferences.get_preference(
            slot, value
        )
        old_count = self._session_slot_value_counts.get(key, 0)

        if old_score is not None and old_count > 0:
            self._update_existing_preference(
                self._session_slot_value_preferences,
                self._session_slot_value_counts,
                slot,
                value,
                score,
            )
            return

        direction = 1 if score > 0 else -1
        new_score = direction * (2 * self.UPDATE_STEP)
        self._session_slot_value_preferences.set_preference(
            slot, value, new_score
        )
        self._session_slot_value_counts[key] = 1

    def _apply_text_update(self, text: str) -> None:
        """Applies preference updates for one user utterance.

        Args:
            text: User utterance.
        """
        text_lower = self._normalize_preference_value(text)
        for slot, value in self._extract_matched_values(text_lower):
            score = self._score_value_in_text(text_lower, value)
            if score != 0:
                self.update_slot_value_preference(slot, value, score)

    def _reset_session_preferences(self) -> None:
        """Clears per-dialogue preference state."""
        self._session_slot_value_preferences = UserPreferences(self._user_id)
        self._session_slot_value_counts.clear()

    def _promote_session_preferences_to_long_term(self) -> None:
        """Promotes well-confirmed session preferences to long-term memory."""
        for (
            slot,
            value_preferences,
        ) in self._session_slot_value_preferences._preferences.items():
            for value, session_score in value_preferences.items():
                key = (slot, value)
                session_count = self._session_slot_value_counts.get(key, 0)
                if session_count < self.LONG_TERM_PROMOTION_MIN_CONFIRMATIONS:
                    continue

                long_term_score = (
                    self._long_term_slot_value_preferences.get_preference(
                        slot, value
                    )
                )
                if long_term_score is None:
                    self._long_term_slot_value_preferences.set_preference(
                        slot, value, session_score
                    )
                    self._long_term_slot_value_counts[key] = session_count
                    continue

                if session_score > long_term_score:
                    new_score = min(
                        long_term_score + self.UPDATE_STEP, session_score
                    )
                else:
                    new_score = max(
                        long_term_score - self.UPDATE_STEP, session_score
                    )
                self._long_term_slot_value_preferences.set_preference(
                    slot, value, new_score
                )
                self._long_term_slot_value_counts[key] = (
                    self._long_term_slot_value_counts.get(key, 0)
                    + session_count
                )

    def _extract_matched_values(self, text_lower: str) -> List[Tuple[str, str]]:
        """Returns matched known or catalog preferences without shorter
        duplicate fragments.

        Args:
            text_lower: Lowercased user utterance.

        Returns:
            Matched slot-value pairs, preferring longer values over nested
            substrings.
        """
        matches: List[Tuple[str, str]] = []
        seen_matches = set()
        for slot, value, _ in self._rank_preferences():
            if len(value) >= 3 and value in text_lower:
                matches.append((slot, value))
                seen_matches.add((slot, value))

        for slot, values in self._catalog_slot_values.items():
            for value in values:
                if (slot, value) in seen_matches:
                    continue
                if value in text_lower:
                    matches.append((slot, value))
                    seen_matches.add((slot, value))

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
        """Updates preferences from one user turn.

        Args:
            dialogue: Dialogue utterance object.
        """
        text = getattr(dialogue, "text", "").strip()
        if not text:
            return

        if text.lower() in self.DIALOGUE_STOP_TOKENS:
            self._promote_session_preferences_to_long_term()
            self._reset_session_preferences()
            return

        self._apply_text_update(text)

    def _rank_preferences(self) -> List[Tuple[str, str, float]]:
        """Ranks session and long-term slot-value preferences.

        Returns:
            Sorted list of slot-value preferences, with session preferences
            overriding long-term ones for the same slot-value pair.
        """
        ranked_preferences: Dict[Tuple[str, str], Tuple[str, str, float]] = {}
        for preference_store in (
            self._long_term_slot_value_preferences,
            self._session_slot_value_preferences,
        ):
            for (
                slot,
                value_preferences,
            ) in preference_store._preferences.items():
                for value, score in value_preferences.items():
                    ranked_preferences[(slot, value)] = (slot, value, score)

        return sorted(
            ranked_preferences.values(),
            key=lambda preference: (
                -math.log1p(
                    self._session_slot_value_counts.get(
                        (preference[0], preference[1]),
                        self._long_term_slot_value_counts.get(
                            (preference[0], preference[1]), 0
                        ),
                    )
                ),
                -abs(preference[2]),
                preference[0],
                preference[1],
            ),
        )

    def _rank_long_term_preferences(self) -> List[Tuple[str, str, float]]:
        """Ranks long-term slot-value preferences by support and strength."""
        ranked_preferences: List[Tuple[str, str, float]] = []
        for (
            slot,
            value_preferences,
        ) in self._long_term_slot_value_preferences._preferences.items():
            for value, score in value_preferences.items():
                ranked_preferences.append((slot, value, score))

        return sorted(
            ranked_preferences,
            key=lambda preference: (
                -math.log1p(
                    self._long_term_slot_value_counts.get(
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
        long_term_preferences = self._rank_long_term_preferences()
        session_preferences = [
            preference
            for preference in self._rank_preferences()
            if self._session_slot_value_preferences.get_preference(
                preference[0], preference[1]
            )
            is not None
        ]
        if not long_term_preferences and not session_preferences:
            return ""

        max_long_term = max(1, max_preferences // 2)
        max_session = max(1, max_preferences - max_long_term)
        sections = [
            (
                "Session preferences",
                session_preferences[:max_session],
            ),
            (
                "Positive preferences",
                [
                    preference
                    for preference in long_term_preferences
                    if preference[2] >= self._preference_threshold
                ][:max_long_term],
            ),
            (
                "Negative preferences",
                [
                    preference
                    for preference in long_term_preferences
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
