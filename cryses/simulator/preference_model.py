"""User preference modeling component.

This class implements the "single item preference" and "personal knowledge
graph" approaches in (Zhang & Balog, KDD'20).

Preferences are stored for (1) items in the collection and (2) slot-value pairs
(for slots defined in the ontology).  Preferences are represented as real values
in [-1,1], where zero corresponds to neutral.
Missing preferences are inferred running time (depending on the model type).
"""

import random
import string
from enum import Enum
from typing import Dict, List

from dialoguekit.core.slot_value_annotation import SlotValueAnnotation
from dialoguekit.core.recsys.ratings import Ratings
from dialoguekit.core.recsys.item_collection import ItemCollection
from dialoguekit.core.intent import Intent
from dialoguekit.core.ontology import Ontology
from dialoguekit.user.user_preferences import UserPreferences


class PreferenceModelVariant(Enum):
    """Corresponds to the different preference model variants in
    (Zhang & Balog, KDD'20).
    """

    SIP = 0  # Single item preference
    PKG = 1  # Personal knowledge graphs


class PreferenceModel:
    """Representation of the user's preferences."""

    def __init__(
        self,
        ontology: Ontology,
        item_collection: ItemCollection,
        historical_ratings: Ratings,
        model_variant: PreferenceModelVariant,
        historical_user_id: str = None,
    ) -> None:
        """Generates a simulated user, by assigning initial preferences based on
        historical ratings according to the specified model type (SIP or PKG).

        Further preferences are inferred along the way as the simulated user is
        being prompted by the agent for preferences.

        Args:
            ontology: Ontology.
            item_collection: Item collection.
            historical_ratings: Historical ratings.
            model_variant: Preference model variant (SIP or PKG).
            historical_user_id (Optional): If provided, the simulated user is
                based on this particular historical user; otherwise, it is based
                on a randomly sampled user. This is mostly added to make the
                class testable.
        """
        self._ontology = ontology
        self._item_collection = item_collection
        self._historical_ratings = historical_ratings
        self._model_variant = model_variant

        # If historical user ID is not provided, randomly pick one.
        self._historical_user_id = (
            historical_user_id or self._historical_ratings.get_random_user_id()
        )
        # Create a random user ID (in {real_user_id}_{3_random_chars} format).
        random_str = "".join(random.choices(string.ascii_uppercase, k=3))
        self._user_id = f"{self._historical_user_id}_{random_str}"

        # Store item and slot-value preferences separately.
        self._item_preferences = UserPreferences(self._user_id)
        self._slotvalue_preferences = UserPreferences(self._user_id)

        # Initialize item preferences by sampling from historical ratings.
        self._initialize_item_preferences()

    def _initialize_item_preferences(self) -> None:
        """Initializes the simulated user's preferences on items by sampling
        from ratings of a historical user."""
        # TODO Update, as currently all historical item preferences are copied;
        # instead, there should only be a sample of them. Note that that'll make
        # testing a bit tricky...
        for item_id, rating in self._historical_ratings.get_user_ratings(
            self._historical_user_id
        ).items():
            self._item_preferences.set_preference("ITEM_ID", item_id, rating)

    def get_item_preference(self, item_id: str) -> float:
        """Returns preference for a given item.

        This is used to answer questions like "Did you like The Fast and the
        Furious?" (assuming the user has seen it).

        Args:
            item_id: Item ID.

        Returns:
            Preference on an item as a float in [-1,1].
        """
        preference = self._item_preferences.get_preference("ITEM_ID", item_id)
        if not preference:
            # TODO Infer and set item preference based on model variant.
            # See: https://github.com/iai-group/cryses/issues/21
            preference = 0
            # Stores inferred preference.
            self._item_preferences.set_preference(
                "ITEM_ID", item_id, preference
            )
        return preference

    def get_slotvalue_preference(self, slot: str, value: str) -> float:
        """Returns preference for a given slot-value pair.

        For example, to answer the question "Do you like action movies?", we'd
        call `get_slotvalue_preference("GENRE", "action").

        Args:
            slot: Slot name.
            value: Slot value.

        Returns:
            Preference on slot-value pair as a float in [-1,1].
        """
        preference = self._slotvalue_preferences.get_preference(slot, value)
        if not preference:
            # TODO Infer and set slot-value preference based on model variant.
            # See: https://github.com/iai-group/cryses/issues/21
            preference = 0
            # Stores inferred preference.
            self._slotvalue_preferences.set_preference(slot, value, preference)
        return preference

    def get_slot_preference(self, slot: str) -> float:
        """Returns preference for a given slot.

        This is used to answer questions like "What kind of movies do you
        like?", where we'd call `get_slot_preference("GENRE").

        Args:
            slot: Slot name.

        Returns:
            Preference on slot as a value-preference pair.
        """
        # Fetch value-preference pairs for a given slot.
        preferences = self._slotvalue_preferences.get_preference(slot)
        if not preferences:
            # TODO Infers and sets slot preference based on model variant.
            # See: https://github.com/iai-group/cryses/issues/21
            pass
        # TODO Need to select which slot value to return if the user has
        # preferences for multiple ones.
        slot = "TDB"
        preference = 0
        return slot, preference

    def get_next_user_slots(
        self, agent_intent: Intent, agent_slot_values: List[SlotValueAnnotation]
    ) -> Dict:
        """Determines the next user slots via loading from the initialized
        preferences or sampling.

        This method is called by the simulated user's NLU.
        """
        # TODO Figure out what could be delegated to NLU, so that this part is
        # kept as simple as possible. Use get_slot_preference() if possible.
        pass
