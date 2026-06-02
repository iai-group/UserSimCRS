from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Tuple

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
    DIALOGUE_STOP_TOKENS = {
        "exit",
        "goodbye",
        "bye",
        "quit",
        "stop",
        "end",
        "giveup",
    }
    SEARCH_GOAL_CUES = {
        "looking for",
        "want a movie",
        "want something",
        "can you recommend",
        "recommend me",
        "i'm after",
        "i am after",
    }
    NEGATIVE_CUES = (
        "don't like",
        "do not like",
        "not like",
        "dislike",
        "hate",
        "avoid",
        "not interested",
    )
    POSITIVE_CUES = ("like", "love", "prefer", "enjoy")

    def __init__(
        self,
        domain: SimulationDomain,
        item_collection: ItemCollection,
        historical_ratings: Ratings,
        historical_user_id: str = None,
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
        self._dialogue_buffer: List[str] = []
        self._initialize_preferences()

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
                yield slot, str(slot_value)

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
        preference = self._slot_value_preferences.get_preference(slot, value)
        return preference if preference is not None else 0

    def update_slot_value_preference(
        self, slot: str, value: str, score: float
    ) -> None:
        """Updates a slot-value preference.

        Args:
            slot: Slot name.
            value: Slot value.
            score: Preference score to store.
        """
        self._assert_slot_exists(slot)
        key = (slot, value)
        old_score = self._slot_value_preferences.get_preference(slot, value)
        old_count = self._slot_value_counts.get(key, 0)
        if old_score is None or old_count == 0:
            new_score = score
        else:
            if score > old_score:
                new_score = min(old_score + self.UPDATE_STEP, score)
            else:
                new_score = max(old_score - self.UPDATE_STEP, score)
        self._slot_value_preferences.set_preference(slot, value, new_score)
        self._slot_value_counts[key] = old_count + 1

    def _apply_text_update(self, text: str) -> None:
        """Applies preference updates for one buffered utterance.

        Args:
            text: User utterance collected during the dialogue.
        """
        text_lower = text.lower()
        global_score = self._score_text(text)
        matched_values = self._extract_matched_values(text_lower)
        if not matched_values:
            return

        has_search_goal_cue = any(
            cue in text_lower for cue in self.SEARCH_GOAL_CUES
        )
        for slot, value in matched_values:
            score = self._score_value_in_text(
                text_lower, value.lower(), global_score, has_search_goal_cue
            )
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
            if len(value) >= 3 and value.lower() in text_lower:
                matches.append((slot, value))

        filtered_matches: List[Tuple[str, str]] = []
        for slot, value in sorted(matches, key=lambda match: -len(match[1])):
            value_lower = value.lower()
            if any(
                value_lower != kept_value.lower()
                and value_lower in kept_value.lower()
                for _, kept_value in filtered_matches
            ):
                continue
            filtered_matches.append((slot, value))
        return filtered_matches

    @staticmethod
    def _score_value_in_text(
        text_lower: str,
        value_lower: str,
        global_score: float,
        has_search_goal_cue: bool,
    ) -> float:
        """Scores one matched value using nearby cues.

        Args:
            text_lower: Lowercased user utterance.
            value_lower: Lowercased slot value being inspected.
            global_score: Fallback score computed for the whole utterance.
            has_search_goal_cue: Whether the utterance looks like a search
              goal rather than a stable preference.

        Returns:
            A local preference score for the matched value.
        """
        contexts = [
            clause.strip()
            for clause in re.split(r"[.!?;]", text_lower)
            if value_lower in clause
        ]
        if not contexts:
            return 0 if has_search_goal_cue else global_score

        negative_prefix_cues = ("avoid", "nothing", "not", "no", "without")
        for context in contexts:
            before_value, _, _ = context.partition(value_lower)
            if any(cue in before_value for cue in negative_prefix_cues):
                return -StructuredPreferenceModel.PREFERENCE_SCORE

        negative_patterns = (
            rf"\btoo much\b[^.?!;]{{0,40}}\b{re.escape(value_lower)}\b",
        )
        for context in contexts:
            if any(
                re.search(pattern, context) for pattern in negative_patterns
            ):
                return -StructuredPreferenceModel.PREFERENCE_SCORE

        positive_patterns = (
            rf"\bi\s+like\b[^.?!;]{{0,20}}\b{re.escape(value_lower)}\b",
            rf"\bi\s+love\b[^.?!;]{{0,20}}\b{re.escape(value_lower)}\b",
            rf"\bi\s+enjoy\b[^.?!;]{{0,20}}\b{re.escape(value_lower)}\b",
            rf"\bi\s+prefer\b[^.?!;]{{0,20}}\b{re.escape(value_lower)}\b",
        )
        for context in contexts:
            if any(
                re.search(pattern, context) for pattern in positive_patterns
            ):
                return StructuredPreferenceModel.PREFERENCE_SCORE

        if has_search_goal_cue:
            return 0
        return global_score

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

    @staticmethod
    def _score_text(text: str) -> float:
        """Maps simple textual preference cues to a preference score.

        Args:
            text: User utterance.

        Returns:
            Positive, negative, or neutral score inferred from coarse cues.
        """
        text_lower = text.lower()
        if any(
            cue in text_lower for cue in StructuredPreferenceModel.NEGATIVE_CUES
        ):
            return -StructuredPreferenceModel.PREFERENCE_SCORE

        if any(
            cue in text_lower
            for cue in StructuredPreferenceModel.SEARCH_GOAL_CUES
        ):
            return 0

        if any(
            cue in text_lower for cue in StructuredPreferenceModel.POSITIVE_CUES
        ):
            return StructuredPreferenceModel.PREFERENCE_SCORE
        return 0

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
                "Ppositive preferences",
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
