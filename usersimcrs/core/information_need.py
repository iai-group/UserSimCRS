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
from typing import Any, Dict, List, Optional

from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item import Item
from usersimcrs.items.item_collection import ItemCollection
from dialoguekit.core.slot_value_annotation import SlotValueAnnotation

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

    The function first identifies a target item aligned with the preference
    model, then samples matching properties from that item as constraints.
    This keeps the constraints grounded in a real item while still biasing
    the information need toward the user's preferences.

    Args:
        domain: Domain knowledge.
        item_collection: Collection of items.
        preference_model: Preference model of the simulated user.

    Returns:
        Information need.
    """
    preferred_constraints: Dict[str, str] = {}
    for slot in domain.get_informable_slots():
        value, score = preference_model.get_slot_preference(slot)
        if value is None or score < preference_model.PREFERENCE_THRESHOLD:
            continue
        preferred_constraints[slot] = value

    target_item = item_collection.get_random_item()
    if preferred_constraints:
        anchor_slot = random.choice(list(preferred_constraints.keys()))
        anchor_value = preferred_constraints[anchor_slot]
        matching_items = item_collection.get_items_by_properties(
            [SlotValueAnnotation(anchor_slot, anchor_value)]
        )
        liked_items = [
            item
            for item in matching_items
            if preference_model.get_item_preference(item.id)
            >= preference_model.PREFERENCE_THRESHOLD
        ]

        if liked_items:
            target_item = random.choice(liked_items)
        elif matching_items:
            target_item = random.choice(matching_items)

    preference_aligned_slots = [
        slot
        for slot, value in preferred_constraints.items()
        if target_item.get_property(slot) == value
    ]

    if preference_aligned_slots:
        num_constraints = random.randint(1, len(preference_aligned_slots))
        sampled_slots = random.sample(preference_aligned_slots, num_constraints)
        constraints = {
            slot: target_item.get_property(slot) for slot in sampled_slots
        }
    else:
        constraints = {}
        informable_slots = set(domain.get_informable_slots()).intersection(
            target_item.properties.keys()
        )
        if informable_slots:
            num_constraints = random.randint(1, len(informable_slots))
            for slot in random.sample(list(informable_slots), num_constraints):
                constraints[slot] = target_item.get_property(slot)

    requestable_slots = [
        slot
        for slot in domain.get_requestable_slots()
        if slot not in constraints
    ]
    if requestable_slots:
        num_requests = random.randint(1, len(requestable_slots))
        requests = random.sample(requestable_slots, num_requests)
    else:
        requests = []

    return InformationNeed([target_item], constraints, requests)


class InformationNeed:
    SLOT_STATE_INCOMPLETE = "incomplete"
    SLOT_STATE_ATTEMPTED = "attempted"
    SLOT_STATE_COMPLETE = "complete"

    def __init__(
        self,
        target_items: List[Item],
        constraints: Dict[str, Any],
        requests: List[str],
        constraint_states: Optional[Dict[str, str]] = None,
        request_states: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initializes an information need.

        Args:
            target_items: Items that the user is interested in.
            constraints: Slot-value pairs representing constraints on the item
              of interest.
            requests: Slots representing the desired information.
            constraint_states: Optional states for constraints.
            request_states: Optional states for requests.
        """
        self.target_items = target_items
        self.constraints = constraints
        self.requested_slots = defaultdict(
            None, {slot: None for slot in requests}
        )
        self.constraint_states = {
            slot: (constraint_states or {}).get(
                slot, self.SLOT_STATE_INCOMPLETE
            )
            for slot in constraints
        }

        self.request_states = {
            slot: (request_states or {}).get(slot, self.SLOT_STATE_INCOMPLETE)
            for slot in requests
        }

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
        self._mark_state(
            self.constraint_states, slot, self.SLOT_STATE_ATTEMPTED
        )

    def mark_constraint_complete(self, slot: str) -> None:
        self._mark_state(self.constraint_states, slot, self.SLOT_STATE_COMPLETE)

    def mark_request_attempted(self, slot: str) -> None:
        self._mark_state(self.request_states, slot, self.SLOT_STATE_ATTEMPTED)

    def mark_request_complete(self, slot: str, value: Any = None) -> None:
        if slot in self.request_states:
            self.requested_slots[slot] = value
            self.request_states[slot] = self.SLOT_STATE_COMPLETE

    def _get_slot_progress(self, states: Dict[str, str]) -> Dict[str, int]:
        """Returns status counts for a single slot category."""
        values = list(states.values())
        return {
            self.SLOT_STATE_INCOMPLETE: values.count(
                self.SLOT_STATE_INCOMPLETE
            ),
            self.SLOT_STATE_ATTEMPTED: values.count(self.SLOT_STATE_ATTEMPTED),
            self.SLOT_STATE_COMPLETE: values.count(self.SLOT_STATE_COMPLETE),
            "total": len(values),
        }

    def get_information_need_progress(self) -> Dict[str, Dict[str, int]]:
        """Returns progress counters for constraints and requests."""
        constraint_progress = self._get_slot_progress(self.constraint_states)
        request_progress = self._get_slot_progress(self.request_states)
        total_states = [
            *self.constraint_states.values(),
            *self.request_states.values(),
        ]
        return {
            "constraints": constraint_progress,
            "requests": request_progress,
            "overall": {
                self.SLOT_STATE_INCOMPLETE: total_states.count(
                    self.SLOT_STATE_INCOMPLETE
                ),
                self.SLOT_STATE_ATTEMPTED: total_states.count(
                    self.SLOT_STATE_ATTEMPTED
                ),
                self.SLOT_STATE_COMPLETE: total_states.count(
                    self.SLOT_STATE_COMPLETE
                ),
                "total": len(total_states),
            },
        }

    def get_information_need_completion_ratio(self) -> float:
        """Returns the fraction of completed information-need components."""
        progress = self.get_information_need_progress()["overall"]
        if progress["total"] == 0:
            return 1.0
        return progress[self.SLOT_STATE_COMPLETE] / progress["total"]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InformationNeed:
        """Creates information need from a dictionary."""
        target_items = [Item(**item) for item in data["target_items"]]
        return cls(
            target_items=target_items,
            constraints=data["constraints"],
            requests=data["requests"],
            constraint_states=data.get("constraint_states"),
            request_states=data.get("request_states"),
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
            "constraint_states": dict(self.constraint_states),
            "request_states": dict(self.request_states),
        }
