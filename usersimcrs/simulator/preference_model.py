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
from typing import Dict, List, Tuple
from collections import defaultdict

from dialoguekit.core.slot_value_annotation import (
    SlotValueAnnotation,
    Annotation,
)
from dialoguekit.core.recsys.ratings import Ratings
from dialoguekit.core.recsys.item_collection import ItemCollection
import dialoguekit.core.intent as Intent
from dialoguekit.core.ontology import Ontology
from dialoguekit.user.user_preferences import UserPreferences
from dialoguekit.core.recsys.item import Item

K_TO_FETCH = 3


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
        self._keyword_preferences = defaultdict(list)

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
            print(item_id)
            for keyword in (
                self._item_collection.get_item(item_id)
                .get_property("keywords")
                .split("|")
            ):
                self._keyword_preferences[keyword].append(rating)

    def _get_property_preferences(
        self, property: str = "genres"
    ) -> Dict[str, float]:
        """Gets the property (e.g, genre) preferences based on rated items.

        This is used to infer both item and slotvalue preference.

        Returns:
            The averaged ratings based on rated items.
        """
        property_ratings = defaultdict(list)
        # Cache all the ratings for properties based on items, e.g.,
        # {"action": [-1, -1, -1]}.
        for (
            hist_item_id,
            rating,
        ) in self._item_preferences.get_preferences("ITEM_ID").items():
            # TODO: handling all properties generally by creating a new issue
            try:
                for prop in (
                    self._item_collection.get_item(hist_item_id).get_property(
                        property
                    )
                    # TODO: remove splitting
                    .split("|")
                ):
                    property_ratings[prop].append(rating)
            except AttributeError:
                print("HEY")
        property_preferences = dict()
        # Calculate the averaged rating for each genre.
        # TODO: use property instead of genre and make property a parameter of
        # property_ratings
        for prop, rating_list in property_ratings.items():
            property_preferences[prop] = round(
                sum(rating_list) / len(rating_list)
            )
        # tester = "testing property preferences."
        return property_preferences

    def get_item_preference(self, item_id: str) -> Tuple[bool, float]:
        """Returns preference for a given item.

        This is used to answer questions like "Did you like The Fast and the
        Furious?" (assuming the user has seen it).

        Args:
            item_id: Item ID.

        Returns:
            If the item has previously been consumed, the boolean will be
            'True'. Preference on an item as a float in [-1,1].
        """
        preference = self._item_preferences.get_preference("ITEM_ID", item_id)
        consumed = True
        if not preference:
            consumed = False
            # Item does not exist or SIP model. In both cases, infer rating.
            if (
                not self._item_collection.exists(item_id)
                or self._model_variant == PreferenceModelVariant.SIP
            ):
                self._item_collection.add_item(Item(item_id, name=item_id))
                preference = random.choice([-1, 1])
            elif self._model_variant == PreferenceModelVariant.PKG:
                # Get current item's properties in a list, e.g., genres.
                current_item_properties = (
                    self._item_collection.get_item(item_id).get_property(
                        "genres"
                    )
                    # TODO: Remove splitting
                    .split("|")
                )
                # Get the properties' perferences based on historically-rated
                # items.
                property_preferences = self._get_property_preferences()

                # Infer the preference based on the cached ratings towards
                # genres.
                averaged_ratings = sum(  # Averaged ratings of all genres.
                    [
                        property_preferences.get(genre, 0)
                        for genre in current_item_properties
                    ]
                ) / len(current_item_properties)
                preference = round(averaged_ratings)
            else:
                raise ValueError("Preference model not supported!")

            # Store inferred preference.
            self._item_preferences.set_preference(
                "ITEM_ID", item_id, preference
            )
        return (consumed, preference)

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
            # Infer and set slot-value preference based on model variant.
            if self._model_variant == PreferenceModelVariant.SIP:
                preference = random.choice([-1, 1])
            elif self._model_variant == PreferenceModelVariant.PKG:
                # Cache all the ratings for properties based on items, e.g.,
                # {"action": [-1, -1, -1]}.
                property_preferences = self._get_property_preferences()

                # Infer the preference towards slot value based on
                # historically-rated items.
                preference = (
                    property_preferences.get(value)
                    if value in property_preferences
                    else random.choice([-1, 1])
                )
            else:
                raise ValueError("Preference model not supported!")

            # Store inferred preference.
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
        preferences = self._slotvalue_preferences.get_preferences(slot)
        if not preferences:
            preferences = self._get_property_preferences()

        # Randomly pick one slot.
        slot = random.choice(list(preferences.keys()))
        preference = preferences.get(slot)
        # TODO: keep track of the slots.
        return slot, preference

    def get_next_user_slots(
        self, agent_intent: Intent, agent_slot_values: List[SlotValueAnnotation]
    ) -> List[Annotation]:
        """Determines the next user slots via loading from the initialized
        preferences or sampling.

        This method is called by the simulated user's NLU.
        """
        # TODO Figure out what could be delegated to NLU, so that this part is
        # kept as simple as possible. Use get_slot_preference() if possible.
        next_user_slots = []
        if not agent_slot_values:
            # Agent is asking for next step?
            pass
        else:
            # Agent is eliciting preference on either genres, keywords or
            # director
            for slot_value in agent_slot_values:
                slot = slot_value._slot
                value = slot_value._value
                # Agent is explicitly asking for which genres the user likes
                if value == "genres":
                    slot = "GENRE"
                    preferences = self._get_property_preferences()
                elif value == "director":
                    slot = "DIRECTOR"
                    preferences = self._get_property_preferences(
                        property="director"
                    )
                elif value == "keywords":
                    slot = "KEYWORD"
                    preferences = self._get_property_preferences(
                        property="keywords"
                    )
                    preferences.update(
                        self._get_property_preferences(property="genres")
                    )
                    # preferences.update(
                    #     self._get_property_preferences(property="director")
                    # )
                sorted_preferences = sorted(
                    preferences.items(), key=lambda item: item[1], reverse=True
                )
                assert len(sorted_preferences) > 0
                top_K_preferences = sorted_preferences[0:K_TO_FETCH]
                sorted_preferences = dict(sorted_preferences)
                for preference in top_K_preferences:
                    next_user_slots.append(
                        Annotation(
                            slot,
                            value=preference[0],
                        )
                    )
        return next_user_slots
