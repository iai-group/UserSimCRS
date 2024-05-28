"""Interface to represent an information need.

The information need comprises three elements: constraints, requests, and
target items. The constraints specify the slot-value pairs that the item of
interest must satisfy, while the requests specify the slots for which the user
wants information. The target items represent the "ground truth" items that the
user is interested in.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List

from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item import Item
from usersimcrs.items.item_collection import ItemCollection


def generate_random_information_need(
    domain: SimulationDomain, item_collection: ItemCollection
) -> InformationNeed:
    """Generates a random information need based on the domain.

    It randomly selects one target item and sets constraints and requests slots.
    The value of constraints are derived from the target's properties. The
    number of constraints and requests are also randomly determined.

    Args:
        domain: Domain knowledge.
        item_collection: Collection of items.

    Returns:
        Information need.
    """
    target_item = random.choice(list(item_collection._items.values()))

    constraints = {}
    informable_slot = set(domain.get_informable_slots()).intersection(
        target_item.properties.keys()
    )
    nb_constraints = random.randint(1, len(informable_slot))
    for slot in random.sample(informable_slot, nb_constraints):
        constraints[slot] = target_item.get_property(slot)

    requestable_slots = set(
        domain.get_requestable_slots()
    ).symmetric_difference(constraints.keys())
    nb_requests = random.randint(1, len(requestable_slots))
    requests = random.sample(requestable_slots, nb_requests)

    return InformationNeed([target_item], constraints, requests)


class InformationNeed:
    def __init__(
        self,
        target_items: List[Item],
        constraints: Dict[str, Any],
        requests: List[str],
    ) -> None:
        """Initializes an information need.

        Args:
            target_items: Items that the user is interested in.
            constraints: Slot-value pairs representing constraints on the item
              of interest.
            requests: Slots representing the desired information.
        """
        self.target_items = target_items
        self.constraints = constraints
        self.requested_slots = defaultdict(
            None, {slot: None for slot in requests}
        )

    def get_constraint_value(self, slot: str) -> Any:
        """Returns the value of a constraint slot.

        Args:
            slot: Slot.

        Returns:
            Value of the slot.
        """
        return self.constraints.get(slot)

    def get_requestable_slots(self) -> List[str]:
        """Returns the list of requestable slots."""
        return [
            slot
            for slot in self.requested_slots
            if not self.requested_slots[slot]
        ]
