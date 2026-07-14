"""Structured preference model backed by historical ratings and metadata."""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

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
    PreferenceUpdateAgent,
    normalize_preference_value,
)


class StructuredPreferenceModel(PreferenceModel):
    """Preference model based on historical ratings and item metadata."""

    UPDATE_STEP = 0.25
    LONG_TERM_PROMOTION_MIN_CONFIRMATIONS = 2
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
            preference_threshold: Minimum absolute score to store as a
              meaningful preference. Defaults to
              `PreferenceModel.PREFERENCE_THRESHOLD`.
        """
        super().__init__(
            domain, item_collection, historical_ratings, historical_user_id
        )
        self._preference_threshold = preference_threshold
        self._item_preferences = UserPreferences(self._user_id)
        self._long_term_preferences = PreferenceMemory(self._user_id)
        self._session_preferences = PreferenceMemory(self._user_id)
        self._signals_extractor = HeuristicPreferenceSignalsExtractor(
            domain, item_collection
        )
        self._update_agent = PreferenceUpdateAgent(
            domain=domain,
            item_collection=item_collection,
            long_term_preferences=self._long_term_preferences,
            session_preferences=self._session_preferences,
            preference_threshold=preference_threshold,
            update_step=self.UPDATE_STEP,
            promotion_min_confirmations=(
                self.LONG_TERM_PROMOTION_MIN_CONFIRMATIONS
            ),
        )
        self._initialize_preferences()

    def _initialize_preferences(self) -> None:
        """Initializes preferences from historical ratings."""
        self._initialize_item_preferences()
        self._initialize_slot_value_preferences()

    def _initialize_item_preferences(self) -> None:
        """Stores strong historical item ratings."""
        user_ratings = self._historical_ratings.get_user_ratings(
            self._historical_user_id
        )
        for item_id, rating in user_ratings.items():
            if abs(rating) >= self._preference_threshold:
                self._item_preferences.set_preference(
                    KEY_ITEM_ID, item_id, rating
                )

    def _initialize_slot_value_preferences(self) -> None:
        """Stores average historical slot-value ratings."""
        slot_value_ratings: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        user_ratings = self._historical_ratings.get_user_ratings(
            self._historical_user_id
        )
        for item_id, rating in user_ratings.items():
            item = self._item_collection.get_item(item_id)
            if item is None:
                continue
            for slot, value in self._iter_preference_values(item):
                slot_value_ratings[slot][value].append(rating)

        for slot, value_ratings in slot_value_ratings.items():
            for value, ratings in value_ratings.items():
                score = sum(ratings) / len(ratings)
                if abs(score) >= self._preference_threshold:
                    self._long_term_preferences.set(
                        slot, value, score, len(ratings)
                    )

    def _iter_preference_values(self, item: Item) -> Iterable[Tuple[str, str]]:
        """Yields normalized preference-relevant slot-value pairs."""
        for slot in self._domain.get_informable_slots():
            if slot.upper() in {"TITLE", "NAME"}:
                continue
            value = item.get_property(slot)
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            for entry in values:
                yield slot, normalize_preference_value(entry)

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
        normalized_value = normalize_preference_value(value)
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
        self._update_agent.update_slot_value_preference(slot, value, score)

    def _rank_preferences(
        self, memory: PreferenceMemory
    ) -> List[Tuple[str, str, float]]:
        """Ranks preferences by evidence count and score.

        Args:
            memory: Preference memory to rank.

        Returns:
            Ranked slot-value preferences.
        """
        return sorted(
            [(slot, value, score) for slot, value, score, _ in memory.items()],
            key=lambda preference: (
                -memory.get_count(preference[0], preference[1]),
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
        session_preferences = self._rank_preferences(self._session_preferences)[
            : max(1, max_preferences // 2)
        ]
        long_term_preferences = self._rank_preferences(
            self._long_term_preferences
        )
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

    def update_from_dialogue(self, dialogue) -> None:
        """Updates preferences from one user turn.

        Args:
            dialogue: Dialogue utterance object.
        """
        text = getattr(dialogue, "text", "").strip()
        if not text:
            return

        if normalize_preference_value(text) in self.DIALOGUE_STOP_TOKENS:
            self._update_agent.end_session()
            return

        self._update_agent.apply(self._signals_extractor.extract(text))
