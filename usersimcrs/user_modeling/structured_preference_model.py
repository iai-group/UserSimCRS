from __future__ import annotations

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
        self._initialize_preferences()

    def _initialize_preferences(self) -> None:
        """Initializes preferences from historical ratings."""
        self._initialize_item_preferences()
        slot_value_ratings = self._collect_slot_value_ratings()
        self._initialize_slot_value_preferences(slot_value_ratings)

    def _initialize_item_preferences(self) -> None:
        """Stores historical item ratings as item preferences."""
        for item_id, rating in self._get_user_ratings().items():
            if abs(rating) >= self._preference_threshold:
                self._item_preferences.set_preference(
                    KEY_ITEM_ID, item_id, rating
                )

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

    def _initialize_slot_value_preferences(
        self, slot_value_ratings: Dict[str, Dict[str, List[float]]]
    ) -> None:
        """Stores average slot-value ratings as slot-value preferences."""
        for slot, value_ratings in slot_value_ratings.items():
            for value, ratings in value_ratings.items():
                score = sum(ratings) / len(ratings)
                if abs(score) >= self._preference_threshold:
                    self._slot_value_preferences.set_preference(
                        slot, value, score
                    )

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
            values = value if isinstance(value, list) else [value]
            for slot_value in values:
                yield slot, str(slot_value)

    def get_item_preference(self, item_id: str) -> float:
        """Returns preference score for an item."""
        self._assert_item_exists(item_id)
        preference = self._item_preferences.get_preference(KEY_ITEM_ID, item_id)
        return preference if preference is not None else 0

    def get_slot_value_preference(self, slot: str, value: str) -> float:
        """Returns preference score for a slot-value pair."""
        self._assert_slot_exists(slot)
        preference = self._slot_value_preferences.get_preference(slot, value)
        return preference if preference is not None else 0
