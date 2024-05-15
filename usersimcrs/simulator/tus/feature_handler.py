"""Feature handler for the TUS simulator.

For a matter of simplicity, the feature handler supports only one domain unlike
the original implementation.
"""

from typing import Any, Dict, List

from dialoguekit.core.dialogue_act import DialogueAct

from usersimcrs.core.information_need import InformationNeed
from usersimcrs.core.simulation_domain import SimulationDomain


class FeatureHandler:
    def __init__(
        self,
        domain: SimulationDomain,
        user_actions: List[str],
        agent_actions: List[str],
    ) -> None:
        """Initializes the feature handler.

        Args:
            domain: Domain knowledge.
            user_actions: User actions.
            agent_actions: Agent actions.
        """
        self._domain = domain
        self._user_actions = user_actions
        self._agent_actions = agent_actions
        self._create_slot_index()

    def _create_slot_index(self) -> Dict[str, int]:
        """Creates an index for slots.

        Returns:
            Slot index.
        """
        slots = set(
            self._domain.get_slot_names()
            + self._domain.get_requestable_slots()
        )
        self.slot_index = {slot: index for index, slot in enumerate(slots)}

    def get_basic_information_feature(
        self,
        slot: str,
        information_need: InformationNeed,
        state: Dict[str, Any],
        previous_state: Dict[str, Any],
    ) -> List[int]:
        """Builds feature vector for basic information.

        It concatenates the value in agent state, user state, slot type,
        completion status, and first mention.

        Args:
            slot: Slot.
            information_need: Information need.
            state: Current state.
            previous_state: Previous state.

        Returns:
            Feature vector for basic information.
        """
        # Represents the value of the slot in the user/agent state.
        # It is a 4-dimensional vector, where each dimension corresponds to the
        # following values: "none", "?", "don't care", and "other values".
        v_user_value = [0] * 4
        if slot not in information_need.constraints.keys():
            v_user_value[0] = 1
        elif (
            slot in information_need.requested_slots.keys()
            and not information_need.requested_slots.get(slot)
        ):
            v_user_value[1] = 1
        elif information_need.get_constraint_value(slot) == "dontcare":
            v_user_value[2] = 1
        else:
            v_user_value[3] = 1

        v_agent_value = [0] * 4
        if slot not in state.keys():
            v_agent_value[0] = 1
        elif slot in state.keys() and not state.get(slot):
            v_agent_value[1] = 1
        elif state.get(slot) == "dontcare":
            v_agent_value[2] = 1
        else:
            v_agent_value[3] = 1

        # Whether or not the slot is a constraint or requestable slot
        v_type = [0, 0]
        if slot in information_need.constraints.keys():
            v_type[0] = 1
        if slot in information_need.requested_slots.keys():
            v_type[1] = 1

        # Whether or not a constraint or informable slot has been fulfilled
        v_ful = (
            [0]
            if not information_need.get_constraint_value(slot)
            or not information_need.get_requestable_slots(slot)
            else [1]
        )

        # Whether or not this is the first mention of the slot
        v_first = [0]
        if slot not in previous_state.keys() and slot in state.keys():
            v_first = [1]

        return v_user_value + v_agent_value + v_type + v_ful + v_first

    def get_agent_action_feature(
        self, agent_dacts: List[DialogueAct]
    ) -> List[int]:
        """Builds feature vector for agent action.

        It concatenates action vectors represented as 3-dimensional vectors that
        describe whether the value is "none", "?", or "other values".

        Args:
            agent_utterance: Agent utterance.

        Returns:
            Feature vector for agent action.
        """
        v_agent_action = {intent: [0] * 3 for intent in self._agent_actions}
        for dact in agent_dacts:
            if dact.intent in self._agent_actions:
                for slot, value in dact.annotations:
                    if slot and value:
                        v_agent_action[dact.intent][2] = 1
                    elif slot and value is None:
                        v_agent_action[dact.intent][1] = 1
            else:
                v_agent_action[dact.intent][0] = 1

    def get_slot_index_feature(self, slot: str) -> List[int]:
        """Builds feature vector for slot index.

        Args:
            slot: Slot.

        Returns:
            Feature vector for slot index.
        """
        v_slot_index = [0] * len(self.slot_index)
        v_slot_index[self.slot_index[slot]] = 1
        return v_slot_index

    def get_slot_feature_vector(
        self,
        slot: str,
        previous_state: Dict[str, Any],
        state: Dict[str, Any],
        information_need: InformationNeed,
        agent_dacts: List[DialogueAct],
        user_action_vector: List[int] = None,
    ) -> List[int]:
        """Builds the feature vector for a slot.

        It concatenate the basic information, user action, agent action, and
        slot index feature vectors.

        Args:
            slot: Slot.
            previous_state: Previous state.
            state: Current state.
            information_need: Information need.
            agent_dacts: Agent dialogue acts.
            user_action_vector: User action feature vector (output vector for
              previous turn). Defaults to None.

        Returns:
            Feature vector for the slot.
        """
        v_user_action = user_action_vector if user_action_vector else [0] * 6
        return (
            self.get_basic_information_feature(
                slot, information_need, state, previous_state
            )
            + v_user_action
            + self.get_agent_action_feature(agent_dacts)
            + self.get_slot_index_feature(slot)
        )

    def get_turn_feature_vector(
        self,
        slots: List[str],
        previous_state: Dict[str, Any],
        state: Dict[str, Any],
        information_need: InformationNeed,
        agent_dacts: List[DialogueAct],
        user_action_vectors: List[List[int]] = None,
    ) -> List[int]:
        """Builds the feature vector for a turn.

        It comprises the feature vectors for all slots.

        Args:
            slots: Slots.
            previous_state: Previous state.
            state: Current state.
            information_need: Information need.
            agent_dacts: Agent dialogue acts.
            user_action_vectors: User action feature vectors. Defaults to None.

        Returns:
            Feature vector for the turn.
        """
        return [
            self.get_slot_feature_vector(
                slot,
                previous_state,
                state,
                information_need,
                agent_dacts,
                user_action_vector=user_action_vector,
            )
            for slot, user_action_vector in zip(slots, user_action_vectors)
        ]
