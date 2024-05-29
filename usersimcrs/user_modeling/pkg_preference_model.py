"""User preference modeling component based on a PKG.

A personal knowledge graph (PKG), in this particular application context, is a
knowledge graph that is used to store the preferences of a single individual to
support the delivery of services that are customized to that individual.

The implementation of get_item_preference() and get_slot_value_preference()
depends on the release of the PKG API.
See: https://github.com/iai-group/UserSimCRS/issues/110
"""

from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.items.ratings import Ratings
from usersimcrs.user_modeling.preference_model import PreferenceModel


class PKGPreferenceModel(PreferenceModel):
    def __init__(
        self,
        domain: SimulationDomain,
        item_collection: ItemCollection,
        historical_ratings: Ratings,
        historical_user_id: str = None,
    ) -> None:
        """Initializes the preference model of a simulated user based on a PKG.

        Args:
            domain: Domain.
            item_collection: Item collection.
            historical_ratings: Historical ratings.
            historical_user_id (Optional): If provided, the simulated user is
                based on this particular historical user; otherwise, it is based
                on a randomly sampled user. This is mostly added to make the
                class testable.
        """
        super().__init__(
            domain, item_collection, historical_ratings, historical_user_id
        )
        # TODO: Open connection to PKG.

    def get_item_preference(self, item_id: str) -> float:
        """Returns a preference for a given item.

        Args:
            item_id: Item ID.

        Returns:
            Item preference, which is in [-1,1].

        Raises:
            ValueError: If the item does not exist in the collection.
        """
        self._assert_item_exists(item_id)
        # TODO: Query PKG to retrieve item preference.
        preference = None
        return preference

    def get_slot_value_preference(self, slot: str, value: str) -> float:
        """Returns a preference on a given slot-value pair.

        Args:
            slot: Slot name (needs to exist in the domain).
            value: Slot value.

        Returns:
            Slot-value preference, which is in [-1,1].
        """
        self._assert_slot_exists(slot)
        # TODO: Query PKG to retrieve slot-value preference.
        preference = None
        return preference
