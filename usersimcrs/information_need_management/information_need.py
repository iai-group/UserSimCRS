"""Interface to represent an information need.

The information need comprises three elements: constraints, requests, and target
items. The constraints specify the slot-value pairs that the item of interest
must satisfy, while the requests specify the slots for which the user wants
information. The target items represent the "ground truth" items that the user
is interested in.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List

from dialoguekit.core.slot_value_annotation import SlotValueAnnotation
from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item import Item
from usersimcrs.items.item_collection import ItemCollection

from usersimcrs.user_modeling.preference_model import PreferenceModel


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
    target_item = item_collection.get_random_item()

    constraints = {}
    informable_slots = set(domain.get_informable_slots()).intersection(
        target_item.properties.keys()
    )
    num_constraints = random.randint(1, len(informable_slots))
    for slot in random.sample(list(informable_slots), num_constraints):
        constraints[slot] = target_item.get_property(slot)

    requestable_slots = set(
        domain.get_requestable_slots()
    ).symmetric_difference(constraints.keys())
    num_requests = random.randint(1, len(requestable_slots))
    requests = random.sample(list(requestable_slots), num_requests)

    return InformationNeed([target_item], constraints, requests)


def generate_preference_grounded_information_need(
    domain: SimulationDomain,
    item_collection: ItemCollection,
    preference_model: PreferenceModel,
) -> InformationNeed:
    """Generates an information need aligned with a preference model.

    The function samples a random number of informable slots, obtains
    preference constraints for those slots, then tries to find an item matching
    them. If no matching item is found, constraints are relaxed until a match is
    found. If no constraints match any items, the information need is returned
    without target items.

    Args:
        domain: Domain knowledge.
        item_collection: Collection of items.
        preference_model: Preference model of the simulated user.

    Returns:
        Information need.
    """
    informable_slots = list(domain.get_informable_slots())
    preferred_slots = random.sample(
        informable_slots, random.randint(1, len(informable_slots))
    )
    preferred_constraints = {
        slot: value
        for slot in preferred_slots
        for value, _ in [preference_model.get_slot_preference(slot)]
        if value is not None
    }

    matching_items = []
    matching_constraints = preferred_constraints
    preferred_constraint_items = list(preferred_constraints.items())
    for num_constraints in range(len(preferred_constraint_items), 0, -1):
        candidate_constraints = dict(
            preferred_constraint_items[:num_constraints]
        )
        matching_items = item_collection.get_items_by_properties(
            [
                SlotValueAnnotation(slot, value)
                for slot, value in candidate_constraints.items()
            ]
        )
        if matching_items:
            matching_constraints = candidate_constraints
            break

    target_items = [random.choice(matching_items)] if matching_items else []

    constraints = (
        {
            slot: value
            for slot in matching_constraints
            for value in [target_items[0].get_property(slot)]
            if value is not None
        }
        if target_items
        else matching_constraints
    )

    requestable_slots = [
        slot
        for slot in domain.get_requestable_slots()
        if slot not in constraints
    ]
    requests = (
        random.sample(
            requestable_slots, random.randint(1, len(requestable_slots))
        )
        if requestable_slots
        else []
    )

    return InformationNeed(target_items, constraints, requests)


class InformationNeed:
    SLOT_STATE_INCOMPLETE = "incomplete"
    SLOT_STATE_ATTEMPTED = "attempted"
    SLOT_STATE_COMPLETE = "complete"

    def __init__(
        self,
        target_items: List[Item],
        constraints: Dict[str, Any],
        requests: List[str],
        constraint_states: Dict[str, str] = {},
        request_states: Dict[str, str] = {},
    ) -> None:
        """Initializes an information need.

        Args:
            target_items: Items that the user is interested in.
            constraints: Slot-value pairs representing constraints on the item
              of interest.
            requests: Slots representing the desired information.
            constraint_states: States for constraints.
            request_states: States for requests.
        """
        self.target_items = target_items
        self.constraints = constraints
        self.requested_slots = defaultdict(
            None, {slot: None for slot in requests}
        )
        self.constraint_states: DefaultDict[str, str] = defaultdict(
            lambda: self.SLOT_STATE_INCOMPLETE,
            {
                slot: constraint_states.get(slot, self.SLOT_STATE_INCOMPLETE)
                for slot in constraints
            },
        )

        self.request_states: DefaultDict[str, str] = defaultdict(
            lambda: self.SLOT_STATE_INCOMPLETE,
            {
                slot: request_states.get(slot, self.SLOT_STATE_INCOMPLETE)
                for slot in requests
            },
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
            if self.request_states.get(slot) != self.SLOT_STATE_COMPLETE
        ]

    def _mark_state(
        self, states: Dict[str, str], slot: str, state: str
    ) -> None:
        """Updates the state of a tracked slot.

        Args:
            states: Mapping from slots to their current states.
            slot: Slot whose state should be updated.
            state: New state to assign to the slot.
        """
        if slot in states:
            states[slot] = state

    def mark_constraint_attempted(self, slot: str) -> None:
        """Marks a constraint slot as attempted."""
        self._mark_state(
            self.constraint_states, slot, self.SLOT_STATE_ATTEMPTED
        )

    def mark_constraint_complete(self, slot: str) -> None:
        """Marks a constraint slot as complete."""
        self._mark_state(self.constraint_states, slot, self.SLOT_STATE_COMPLETE)

    def mark_request_attempted(self, slot: str) -> None:
        """Marks a request slot as attempted."""
        self._mark_state(self.request_states, slot, self.SLOT_STATE_ATTEMPTED)

    def mark_request_complete(self, slot: str, value: Any) -> None:
        """Marks a request slot as complete with the observed value."""
        if slot in self.request_states:
            self.requested_slots[slot] = value
            self.request_states[slot] = self.SLOT_STATE_COMPLETE

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InformationNeed:
        """Creates information need from a dictionary."""
        target_items = [Item(**item) for item in data["target_items"]]
        return cls(
            target_items=target_items,
            constraints=data["constraints"],
            requests=data["requests"],
            constraint_states=data.get("constraint_states", {}),
            request_states=data.get("request_states", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Returns information need as a dictionary."""
        return {
            "target_items": [
                {"item_id": item.id, "properties": item.properties}
                for item in self.target_items
            ],
            "constraints": self.constraints,
            "requests": list(self.requested_slots.keys()),
            "constraint_states": self.constraint_states,
            "request_states": self.request_states,
        }
