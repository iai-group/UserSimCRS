"""Feature handler for the TUS simulator.

For a matter of simplicity, the feature handler supports only one domain unlike
the original implementation.
"""

from typing import Dict, List, Set

import torch
from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue_act import DialogueAct

from usersimcrs.core.information_need import InformationNeed
from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.dialogue_management.dialogue_state import DialogueState
from usersimcrs.simulator.neural_based.core.feature_handler import (
    FeatureHandler,
)


class TUSFeatureHandler(FeatureHandler):
    def __init__(
        self,
        domain: SimulationDomain,
        agent_actions: List[str],
        user_actions: List[str] = ["inform", "request"],
    ) -> None:
        """Initializes the feature handler.

        Args:
            domain: Domain knowledge.
            agent_actions: Agent actions.
            user_actions: User actions. Defaults to ["inform", "request"].
        """
        self._domain = domain
        self._user_actions = user_actions
        self._agent_actions = agent_actions
        self.action_slots: Set[str] = set()
        self._create_slot_index()

    def reset(self) -> None:
        """Resets the feature handler."""
        self.action_slots = set()

    def _create_slot_index(self) -> None:
        """Creates an index for slots."""
        slots = set(
            self._domain.get_slot_names() + self._domain.get_requestable_slots()
        )
        self.slot_index = {slot: index for index, slot in enumerate(slots)}

    def get_basic_information_feature(
        self,
        slot: str,
        information_need: InformationNeed,
        state: DialogueState,
        previous_state: DialogueState,
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
        if (
            slot not in information_need.constraints.keys()
            and slot not in information_need.requested_slots.keys()
        ):
            v_user_value[0] = 1
        elif (
            slot in information_need.requested_slots.keys()
            and information_need.requested_slots.get(slot) is None
        ):
            v_user_value[1] = 1
        elif information_need.get_constraint_value(slot) == "dontcare":
            v_user_value[2] = 1
        else:
            v_user_value[3] = 1

        v_agent_value = [0] * 4
        if slot not in state.belief_state.keys():
            v_agent_value[0] = 1
        elif (
            slot in state.belief_state.keys()
            and state.belief_state.get(slot) is None
        ):
            v_agent_value[1] = 1
        elif state.belief_state.get(slot) == "dontcare":
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
            [1]
            if (
                state.belief_state.get(slot) is not None
                and state.belief_state.get(slot)
                == information_need.get_constraint_value(slot)
            )
            or information_need.requested_slots.get(slot)
            else [0]
        )

        # Whether or not this is the first mention of the slot
        v_first = [0]
        if (
            slot not in previous_state.belief_state.keys()
            and slot in state.belief_state.keys()
        ):
            v_first = [1]

        return v_user_value + v_agent_value + v_type + v_ful + v_first

    def get_agent_action_feature(
        self, agent_dacts: List[DialogueAct]
    ) -> List[int]:
        """Builds feature vector for agent action.

        It concatenates action vectors represented as 3-dimensional vectors that
        describe whether the slot and value are absent, only slot is present, or
        both slot and value are present.

        Args:
            agent_dacts: Agent dialogue acts.

        Returns:
            Feature vector for agent action.
        """
        v_agent_action = {intent: [0] * 3 for intent in self._agent_actions}
        for agent_dact in agent_dacts:
            if agent_dact.intent in self._agent_actions:
                if not agent_dact.annotations:
                    v_agent_action[agent_dact.intent][0] = 1
                for annotation in agent_dact.annotations:
                    if annotation.slot and annotation.value:
                        v_agent_action[agent_dact.intent][2] = 1
                    elif annotation.slot and annotation.value is None:
                        v_agent_action[agent_dact.intent][1] = 1
        return sum(v_agent_action.values(), [])

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
        previous_state: DialogueState,
        state: DialogueState,
        information_need: InformationNeed,
        agent_dacts: List[DialogueAct],
        user_action_vector: torch.Tensor = None,
    ) -> torch.Tensor:
        """Builds the feature vector for a slot.

        It concatenate the basic information, user action, agent action, and
        slot index feature vectors.

        Args:
            slot: Slot.
            previous_state: Previous state.
            state: Current state.
            information_need: Information need.
            agent_dact: Agent dialogue acts.
            user_action_vector: User action feature vector (output vector for
              previous turn). Defaults to None.

        Returns:
            Feature vector for the slot.
        """
        v_user_action = (
            user_action_vector
            if user_action_vector is not None
            else torch.tensor([0] * 6)
        )
        agent_dacts = []
        for dact in agent_dacts:
            for annotation in dact.annotations:
                if annotation.slot == slot or annotation.slot is None:
                    agent_dacts.append(dact)

        return torch.tensor(
            self.get_basic_information_feature(
                slot, information_need, state, previous_state
            )
            + self.get_agent_action_feature(agent_dacts)
            + v_user_action.tolist()
            + self.get_slot_index_feature(slot)
        )

    def get_feature_vector(
        self,
        utterance: AnnotatedUtterance,
        previous_state: DialogueState,
        state: DialogueState,
        information_need: InformationNeed,
        user_action_vectors: Dict[str, torch.Tensor] = {},
    ) -> torch.Tensor:
        """Builds the feature vector for a turn.

        It comprises the feature vectors for all slots that in the
        information need and mentioned during the conversation.

        Args:
            utterance: Agent utterance with annotations.
            slots: Slots.
            previous_state: Previous state.
            state: Current state.
            information_need: Information need.
            user_action_vectors: User action feature vectors per slot. Defaults
              to an empty dictionary.

        Returns:
            Feature vector for the turn.
        """
        try:
            agent_dacts = utterance.dialogue_acts
        except AttributeError:
            agent_dacts = [DialogueAct(utterance.intent, utterance.annotations)]

        self.action_slots.update(
            [
                annotation.slot
                for dact in agent_dacts
                for annotation in dact.annotations
            ]
            + list(information_need.constraints.keys())
            + list(information_need.requested_slots.keys())
        )
        return torch.cat(
            [
                self.get_slot_feature_vector(
                    slot,
                    previous_state,
                    state,
                    information_need,
                    agent_dacts,
                    user_action_vectors.get(slot, None),
                )
                for slot in self.action_slots
            ],
        )

    def get_label_vector(
        self,
        user_utterance: AnnotatedUtterance,
        current_state: DialogueState,
        information_need: InformationNeed,
    ) -> torch.Tensor:
        """Builds the label vector for a turn.

        It comprises a one-hot encoded vector that determines the value of each
        slot.

        Args:
            user_utterance: User utterance with annotations.
            current_state: Current state.
            information_need: Information need.

        Returns:
            Label vector for the turn.
        """
        try:
            user_dacts = user_utterance.dialogue_acts
        except AttributeError:
            user_dacts = [
                DialogueAct(user_utterance.intent, user_utterance.annotations)
            ]

        output = []
        for slot in self.action_slots:
            o = self._get_label_vector_slot(
                user_dacts, slot, current_state, information_need
            )
            output.append(o)

        return torch.tensor(output)

    def _get_label_vector_slot(
        self,
        user_dacts: List[DialogueAct],
        slot: str,
        current_state: DialogueState,
        information_need: InformationNeed,
    ):
        """Builds the label vector for a slot.

        It is a 6-dimensional vector, where each dimension corresponds to the
        following values: "none", "don't care", "?", "from information need",
        "from belief state", and "random".

        Args:
            user_dacts: User dialogue acts.
            slot: Slot.
            current_state: Current state.
            information_need: Information need.

        Returns:
            Label vector for the slot.
        """
        o = [0] * 6
        for dact in user_dacts:
            for annotation in dact.annotations:
                if annotation.slot == slot:
                    if annotation.value == "dontcare":
                        o[1] = 1
                    elif annotation.value is None:
                        # The value is requested by the user
                        o[2] = 1
                    elif (
                        annotation.value
                        == information_need.get_constraint_value(slot)
                        or annotation.value
                        == information_need.requested_slots.get(slot)
                    ):
                        # The value is taken from the information need
                        o[3] = 1
                    elif annotation.value == current_state.belief_state.get(
                        slot
                    ):
                        # The value was previously mentioned and is
                        # retrieved from the belief state
                        o[4] = 1
                    else:
                        # The slot's value is randomly chosen
                        o[5] = 1

        if o == [0] * 6:
            # The slot is not mentioned in the user utterance
            o[0] = 1
        return o
