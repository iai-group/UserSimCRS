"""Interface to represent an information need.

The information need is divided into two parts: constraints and requests. The
constraints specify the slot-value pairs that the item of interest must satisfy,
while the requests specify the slots for which the user wants information.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List

from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item_collection import ItemCollection

random.seed(42)


def generate_information_need(
    domain: SimulationDomain, item_collection: ItemCollection
) -> InformationNeed:
    """Generates an information need based on the domain.

    Args:
        domain: Domain knowledge.
        item_collection: Collection of items.

    Returns:
        Information need.
    """
    slot_names = domain.get_slot_names()
    constraints = {}
    nb_constraints = random.randint(1, len(slot_names))
    for s in random.sample(slot_names, nb_constraints):
        value = random.choice(
            list(item_collection.get_possible_property_values(s))
        )
        constraints[s] = value

    requestable_slots = domain.get_requestable_slots()
    nb_requests = random.randint(1, len(requestable_slots))
    requests = random.sample(requestable_slots, nb_requests)

    return InformationNeed(constraints, requests)


class InformationNeed:
    def __init__(
        self, constraints: Dict[str, Any], requests: List[str]
    ) -> None:
        """Initializes an information need.

        Args:
            constraints: Slot-value pairs representing constraints on the item
              of interest.
            requests: Slots representing the desired information.
        """
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
