from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.items.ratings import Ratings
from usersimcrs.user_modeling.preference_model import PreferenceModel


class StructuredPreferenceModel(PreferenceModel):
    """Preference model based on historical ratings and item metadata."""

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
        self._preferences: Dict[str, Dict[str, float]] = {}
        self._initialize_preferences()

    def _initialize_preferences(self) -> None:
        """Initializes preferences from historical ratings."""
        slot_value_ratings = self._collect_slot_value_ratings()
        self._preferences = self._average_slot_value_ratings(slot_value_ratings)

    def _collect_slot_value_ratings(self) -> Dict[str, Dict[str, List[float]]]:
        """Collects historical ratings for each slot-value pair."""
        slot_value_ratings: Dict[str, Dict[str, List[float]]] = {}
        for item_id, rating in self._get_user_ratings().items():
            item = self._item_collection.get_item(item_id)
            if item is None:
                continue

            for slot, value in self._iter_preference_values(item):
                slot_value_ratings.setdefault(slot, {}).setdefault(
                    value, []
                ).append(rating)
        return slot_value_ratings

    def _average_slot_value_ratings(
        self, slot_value_ratings: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Dict[str, float]]:
        """Converts collected ratings into average preference scores."""
        preferences: Dict[str, Dict[str, float]] = {}
        for slot, value_ratings in slot_value_ratings.items():
            for value, ratings in value_ratings.items():
                score = sum(ratings) / len(ratings)
                if abs(score) >= self._preference_threshold:
                    preferences.setdefault(slot, {})[value] = score
        return preferences

    def _get_user_ratings(self) -> Dict[str, float]:
        """Returns historical ratings for the selected user."""
        return self._historical_ratings.get_user_ratings(
            self._historical_user_id
        )

    def _iter_preference_values(self, item) -> Iterable[Tuple[str, str]]:
        """Yields preference-relevant slot-value pairs for an item."""
        for slot in self._domain.get_slot_names():
            if slot.upper() in {"TITLE", "NAME"}:
                continue
            value = item.get_property(slot)
            if value is None:
                continue
            for normalized_value in self._normalize_values(value):
                yield self._normalize_key(slot), self._normalize_key(
                    normalized_value
                )

    @staticmethod
    def _normalize_values(value: Any) -> List[str]:
        """Normalizes single and multi-value item properties."""
        if isinstance(value, list):
            return [str(v) for v in value if str(v)]
        return [str(value)]

    @staticmethod
    def _normalize_key(value: str) -> str:
        """Normalizes preference dictionary keys."""
        return str(value).strip().lower()

    def get_item_preference(self, item_id: str) -> float:
        """Returns preference score inferred from the item's slot values."""
        self._assert_item_exists(item_id)
        item = self._item_collection.get_item(item_id)
        if item is None:
            return 0

        scores = [
            self._preferences[slot][value]
            for slot, value in self._iter_preference_values(item)
            if value in self._preferences.get(slot, {})
        ]
        return sum(scores) / len(scores) if scores else 0

    def get_slot_value_preference(self, slot: str, value: str) -> float:
        """Returns preference score for a slot-value pair."""
        self._assert_slot_exists(slot)
        slot_key = self._normalize_key(slot)
        value_key = self._normalize_key(value)
        return self._preferences.get(slot_key, {}).get(value_key, 0)

    def get_slot_preference(self, slot: str) -> Tuple[str, float]:
        """Returns the strongest positive slot-value preference."""
        self._assert_slot_exists(slot)
        slot_key = self._normalize_key(slot)
        positive_preferences = [
            (value, score)
            for value, score in self._preferences.get(slot_key, {}).items()
            if score > self.PREFERENCE_THRESHOLD
        ]
        if not positive_preferences:
            return None, 0
        return max(positive_preferences, key=lambda item: item[1])
