"""User preference modeling interface.

Preferences are stored for (1) items in the collection and (2) slot-value pairs
(for slots defined in the domain).  Preferences are represented as real values
in [-1,1], where zero corresponds to neutral.
"""

import random
import string
from abc import ABC, abstractmethod
from typing import Tuple

from dialoguekit.core.domain import Domain

from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.items.ratings import Ratings

# Key used to identify items. This will need to be replaced once support for
# multiple entity types is added.
KEY_ITEM_ID = "ITEM_ID"


class PreferenceModel(ABC):
    """Representation of the user's preferences."""

    # Above this threshold, a preference is considered positive;
    # below -1 x this threshold, a preference is considered negative;
    # otherwise, it's considered neutral.
    PREFERENCE_THRESHOLD = 0.25

    def __init__(
        self,
        domain: Domain,
        item_collection: ItemCollection,
        historical_ratings: Ratings,
        historical_user_id: str = None,
    ) -> None:
        """Initializes the preference model of a simulated user.

        A list of initial seen items is generated based on historical ratings.
        Further preferences are inferred along the way as the simulated user is
        being prompted by the agent for preferences.

        Args:
            domain: Domain.
            item_collection: Item collection.
            historical_ratings: Historical ratings.
            historical_user_id (Optional): If provided, the simulated user is
              based on this particular historical user; otherwise, it is based
              on a randomly sampled user. This is mostly added to make the
              class testable. Defaults to None.
        """
        self._domain = domain
        self._item_collection = item_collection
        self._historical_ratings = historical_ratings

        # If historical user ID is not provided, randomly pick one.
        self._historical_user_id = (
            historical_user_id or self._historical_ratings.get_random_user_id()
        )
        # Create a random user ID (in {real_user_id}_{3_random_chars} format).
        random_str = "".join(random.choices(string.ascii_uppercase, k=3))
        self._user_id = f"{self._historical_user_id}_{random_str}"

    def is_item_consumed(self, item_id: str) -> bool:
        """Returns whether or not an item has been consumed by the user.

        This is used to answer questions like: "Have you seen Inception?"

        Args:
            item_id: Item ID.

        Returns:
            True if the item has been consumed (i.e., appears among the
            historical ratings).
        """
        return (
            self._historical_ratings.get_user_item_rating(
                self._historical_user_id, item_id
            )
            is not None
        )

    def _assert_item_exists(self, item_id: str) -> None:
        """Checks if item exists in the collection and throws an exception if
        not."""
        if not self._item_collection.exists(item_id):
            raise ValueError("Item does not exist in item collection.")

    def _assert_slot_exists(self, slot: str) -> None:
        """Checks if slot exists in the domain and throws an exception if
        not."""
        if slot not in self._domain.get_slot_names():
            raise ValueError(
                f"The slot '{slot}' does not exist in this domain."
            )

    @abstractmethod
    def get_item_preference(self, item_id: str) -> float:
        """Returns a preference for a given item.

        This is used to answer questions like: "How did you like it?",
        where "it" refers to the movie mentioned previously.

        Args:
            item_id: Item ID.

        Returns:
            Item preference, which is generally in [-1,1].

        Raises:
            NotImplementedError: If not implemented in derived class.
        """
        raise NotImplementedError

    @abstractmethod
    def get_slot_value_preference(self, slot: str, value: str) -> float:
        """Returns a preference on a given slot-value pair.

        This is used to answer questions like: "Do you like action movies?"

        Args:
            slot: Slot name (needs to exist in the domain).
            value: Slot value.

        Returns:
            Slot-value preference.

        Raises:
            NotImplementedError: If not implemented in derived class.
        """
        raise NotImplementedError

    def get_slot_preference(self, slot: str) -> Tuple[str, float]:
        """Returns a preferred value for a given slot.

        This is used to answer questions like: "What movie genre do you prefer?"

        While in principle negative preferences could also be returned, here it
        is always a positive preference that is expressed.

        Args:
            slot: Slot name (needs to exist in the domain).

        Returns:
            A value and corresponding preferences; if no preference could be
            obtained for that slot, then (None, 0) are returned.
        """
        self._assert_slot_exists(slot)
        preference = None
        attempts = 0

        while not preference:
            # Pick a random value for the slot.
            value: str = random.choice(
                list(self._item_collection.get_possible_property_values(slot))
            )

            preference = self.get_slot_value_preference(slot, value)
            if preference < self.PREFERENCE_THRESHOLD:
                preference = None

            # It would in principle be possible to enter into an infinite loop
            # here (e.g., if there is a small set of possible values for the
            # slot and the user has already expressed negative preference on all
            # of them), therefore we limit the number of attempts.
            attempts += 1
            if attempts == 10:
                return None, 0

        return value, preference
