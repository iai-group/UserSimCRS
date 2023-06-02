"""Simple user preference modeling component.

This class implements the "single item preference" approach in [Zhang & Balog,
KDD'20].

- Whenever a user is prompted whether they had seen/consumed a specific item, we
  answer that based on historical data.
- Whenever the user is prompted for their preference on a given item or
  slot-value pair, we provide a positive/negative response by flipping a coin
  (i.e., either -1 or +1 as the preference).
- Whenever the user is prompted for a preference on a given slot, a random value
  among the existing slot values is picked and returned as positive preference.

The responses given are remembered so that the user would respond the same way
if they are asked the same question again.

This approach offers limited consistency. Items that are seen/consumed are
rooted in real user behavior, but the preferences expressed about them are not.
Hence, the user might express a preference about a slot that is inconsistent
with the answers given to other questions (e.g., likes "action" as a genre, but
has not seen a single action movie).
"""

import random

from dialoguekit.core.domain import Domain
from dialoguekit.participant.user_preferences import UserPreferences

from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.items.ratings import Ratings
from usersimcrs.user_modeling.preference_model import (
    KEY_ITEM_ID,
    PreferenceModel,
)


class SimplePreferenceModel(PreferenceModel):
    def __init__(
        self,
        domain: Domain,
        item_collection: ItemCollection,
        historical_ratings: Ratings,
        historical_user_id: str = None,
    ) -> None:
        """Initializes the simple preference model of a simulated user.

        Args:
            domain: Domain.
            item_collection: Item collection.
            historical_ratings: Historical ratings.
            historical_user_id (Optional): If provided, the simulated user is
              based on this particular historical user; otherwise, it is based
              on a randomly sampled user. This is mostly added to make the
              class testable. Defaults to None.
        """
        super().__init__(
            domain, item_collection, historical_ratings, historical_user_id
        )

        # Store item and slot-value preferences separately.
        self._item_preferences = UserPreferences(self._user_id)
        self._slot_value_preferences = UserPreferences(self._user_id)

    def get_item_preference(self, item_id: str) -> float:
        """Returns a preference for a given item.

        Args:
            item_id: Item ID.

        Returns:
            Randomly chosen preference, which is either -1 or +1.

        Raises:
            ValueError: If the item does not exist in the collection.
        """
        self._assert_item_exists(item_id)
        preference = self._item_preferences.get_preference(KEY_ITEM_ID, item_id)
        if not preference:
            preference = random.choice([-1, 1])
            self._item_preferences.set_preference(
                KEY_ITEM_ID, item_id, preference
            )
        return preference

    def get_slot_value_preference(self, slot: str, value: str) -> float:
        """Returns a preference on a given slot-value pair.

        Args:
            slot: Slot name (needs to exist in the domain).
            value: Slot value.

        Returns:
            Randomly chosen preference, which is either -1 or +1.
        """
        self._assert_slot_exists(slot)
        preference = self._slot_value_preferences.get_preference(slot, value)
        if not preference:
            preference = random.choice([-1, 1])
            self._slot_value_preferences.set_preference(slot, value, preference)
        return preference
