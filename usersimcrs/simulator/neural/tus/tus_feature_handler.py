"""Feature handler for the TUS simulator.

For simplicity, the feature handler supports a single domain unlike the original
implementation.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, List, Tuple

import joblib
import torch

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.slot_value_annotation import SlotValueAnnotation
from usersimcrs.core.information_need import InformationNeed
from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.dialogue_management.dialogue_state import DialogueState
from usersimcrs.simulator.neural.core.feature_handler import (
    FeatureHandler,
    FeatureMask,
    FeatureVector,
)


class TUSFeatureHandler(FeatureHandler):
    def __init__(
        self,
        domain: SimulationDomain,
        max_turn_feature_length: int,
        context_depth: int,
        agent_actions: List[str],
        user_actions: List[str] = ["inform", "request"],
    ) -> None:
        """Initializes the feature handler.

        Args:
            domain: Domain knowledge.
            max_turn_feature_length: Maximum length of a turn feature vector.
            context_depth: Number of previous turns to include in the input
              vector.
            agent_actions: Agent actions.
            user_actions: User actions. Defaults to ["inform", "request"].
        """
        self._domain = domain
        self.max_turn_feature_length = max_turn_feature_length
        self.context_depth = context_depth
        self._user_actions = user_actions
        self._agent_actions = agent_actions
        self.action_slots: List[str] = list()
        self._create_slot_index()
        # Store user feature vectors for each turn
        self.user_feature_history: List[List[FeatureVector]] = list()

    def reset(self) -> None:
        """Resets the feature handler."""
        self.reset_user_feature_history()
        self.action_slots = list()

    def reset_user_feature_history(self) -> None:
        """Resets the user feature history."""
        self.user_feature_history = list()

    def update_action_slots(self, slots: Iterable[str]) -> None:
        """Updates the action slots.

        Action slots are slots present in the information need and mentioned
        during the conversation.

        Args:
            slots: Slots mentioned in an utterance.
        """
        for slot in slots:
            if slot not in self.action_slots:
                self.action_slots.append(slot)

    def _create_slot_index(self) -> None:
        """Creates an index for slots."""
        slots = self._domain.get_slot_names()
        self.slot_index = {slot: index for index, slot in enumerate(slots)}

    def get_basic_information_feature(
        self,
        slot: str,
        information_need: InformationNeed,
        state: DialogueState,
        previous_state: DialogueState,
    ) -> FeatureVector:
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
        self, agent_dialogue_acts: List[DialogueAct]
    ) -> FeatureVector:
        """Builds feature vector for agent action.

        It concatenates action vectors represented as 3-dimensional vectors that
        describe whether the slot and value are absent, only slot is present, or
        both slot and value are present.

        Args:
            agent_dialogue_acts: Agent dialogue acts.

        Returns:
            Feature vector for agent action.
        """
        v_agent_action = {intent: [0] * 3 for intent in self._agent_actions}
        for dialogue_act in agent_dialogue_acts:
            intent_label = dialogue_act.intent.label
            if intent_label in self._agent_actions:
                if not dialogue_act.annotations:
                    v_agent_action[intent_label][0] = 1
                for annotation in dialogue_act.annotations:
                    if annotation.slot and annotation.value:
                        v_agent_action[intent_label][2] = 1
                    elif annotation.slot and annotation.value is None:
                        v_agent_action[intent_label][1] = 1
        return sum(v_agent_action.values(), [])

    def get_slot_index_feature(self, slot: str) -> FeatureVector:
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
        agent_dialogue_acts: List[DialogueAct],
        user_action_vector: torch.Tensor = None,
    ) -> FeatureVector:
        """Builds the feature vector for a slot.

        It concatenates the basic information, user action, agent action, and
        slot index feature vectors.

        Args:
            slot: Slot.
            previous_state: Previous state.
            state: Current state.
            information_need: Information need.
            agent_dialogue_acts: Agent dialogue acts.
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
        _agent_dialogue_acts = []
        # Filter agent dialogue acts by slot
        for dialogue_act in agent_dialogue_acts:
            for annotation in dialogue_act.annotations:
                if annotation.slot == slot or annotation.slot is None:
                    _agent_dialogue_acts.append(dialogue_act)

        return (
            [0, 0]  # No special token
            + self.get_basic_information_feature(
                slot, information_need, state, previous_state
            )
            + self.get_agent_action_feature(_agent_dialogue_acts)
            + v_user_action.tolist()
            + self.get_slot_index_feature(slot)
        )

    def get_turn_feature_vectors(
        self,
        agent_dialogue_acts: List[DialogueAct],
        previous_state: DialogueState,
        state: DialogueState,
        information_need: InformationNeed,
        user_action_vectors: Dict[str, torch.Tensor] = {},
    ) -> List[FeatureVector]:
        """Builds the feature vectors for a turn.

        It comprises the feature vectors for all slots that in the
        information need and mentioned during the conversation.

        Args:
            agent_dialogue_acts: Agent dialogue acts.
            previous_state: Previous state.
            state: Current state.
            information_need: Information need.
            user_action_vectors: User action feature vectors per slot. Defaults
              to an empty dictionary.

        Returns:
            Feature vectors for the turn.
        """

        # Update the action slots with constraints, requested, and mentioned
        # slots
        self.update_action_slots(information_need.constraints.keys())
        self.update_action_slots(information_need.requested_slots.keys())
        self.update_action_slots(
            [
                annotation.slot
                for dialogue_act in agent_dialogue_acts
                for annotation in dialogue_act.annotations
            ]
        )
        return [
            self.get_slot_feature_vector(
                slot,
                previous_state,
                state,
                information_need,
                agent_dialogue_acts,
                user_action_vectors.get(slot, None),
            )
            for slot in self.action_slots
        ]

    def _get_special_token_feature_vector(self, token: str) -> FeatureVector:
        """Builds the feature vector for a special token.

        Args:
            token: Special token, either "[CLS]" or "[SEP]".

        Raises:
            ValueError: If the token is not supported.

        Returns:
            Feature vector for the special token.
        """
        if token not in ["[CLS]", "[SEP]"]:
            raise ValueError(
                f"Unsupported special token: {token}. Supported tokens: [CLS], "
                "[SEP]"
            )

        vector = [1, 0] if token == "[CLS]" else [0, 1]
        vector += [0] * 12  # Dimension of basic information feature
        vector += [0] * len(self.slot_index)  # Dimension of slot index feature
        vector += (
            [0] * 3 * len(self._agent_actions)
        )  # Dimension of agent action feature
        vector += [0] * 6  # Dimension of user action feature
        return vector

    def build_input_vector(
        self,
        agent_dialogue_acts: List[DialogueAct],
        previous_state: DialogueState,
        state: DialogueState,
        information_need: InformationNeed,
        user_action_vectors: Dict[str, torch.Tensor] = {},
    ) -> Tuple[List[FeatureVector], FeatureMask]:
        """Builds the input vector $V_{input}$ for a turn.

        It concatenates the feature vectors for the last n turns separated by
        a special token. Note that is inferred from the list of feature vectors
        provided.
        The input vector is structured as follows:
        [CLS] $V^t$ [SEP] $V^{t-1}$ [SEP] ... [SEP] $V^{t-n}$ [SEP] {padding},
        where {padding} indicates the padding to reach the maximum length.

        Args:
            agent_dialogue_acts: Agent dialogue acts.
            previous_state: Previous state.
            state: Current state.
            information_need: Information need.
            user_action_vectors: User action feature vectors per slot. Defaults
              to an empty dictionary.

        Returns:
            Input vector.
        """
        current_turn_feature_vectors = self.get_turn_feature_vectors(
            agent_dialogue_acts,
            previous_state,
            state,
            information_need,
            user_action_vectors,
        )
        self.user_feature_history.append(current_turn_feature_vectors)

        v_cls = self._get_special_token_feature_vector("[CLS]")
        v_sep = self._get_special_token_feature_vector("[SEP]")
        input_vector: List[FeatureVector] = [v_cls]
        feature_dimension = len(v_cls)
        for turn_feature_vector in reversed(
            self.user_feature_history[-self.context_depth :]  # noqa: E203
        ):
            input_vector.extend(turn_feature_vector)
            input_vector.append(v_sep)
        # input_vector: torch.Tensor = torch.cat(input_vector)

        # Pad the input vector and create mask
        max_length = self.max_turn_feature_length * self.context_depth
        if len(input_vector) < max_length:
            padding = [[0] * feature_dimension] * (
                max_length - len(input_vector)
            )
            input_vector += padding
            mask = [False] * len(input_vector) + [True] * (
                max_length - len(input_vector)
            )
        else:
            mask = [False] * max_length
        return input_vector[:max_length], mask[:max_length]

    def get_label_vector(
        self,
        user_utterance: AnnotatedUtterance,
        current_state: DialogueState,
        information_need: InformationNeed,
    ) -> FeatureVector:
        """Builds the label vector for a turn.

        Args:
            user_utterance: User utterance with annotations.
            current_state: Current state.
            information_need: Information need.

        Returns:
            Label vector for the turn.
        """
        user_dialogue_acts = user_utterance.dialogue_acts
        output = [-1] * self.max_turn_feature_length
        for dialogue_act in user_dialogue_acts:
            for annotation in dialogue_act.annotations:
                if annotation.slot not in self.action_slots:
                    continue
                slot_index = self.action_slots.index(annotation.slot)
                if slot_index >= self.max_turn_feature_length:
                    continue
                label = self._get_label(
                    annotation, current_state, information_need
                )
                output[slot_index] = label

        for i in range(len(self.action_slots)):
            if i < self.max_turn_feature_length and output[i] == -1:
                # The slot is not mentioned in the user utterance
                output[i] = 0

        return output

    def _get_label(
        self,
        annotation: SlotValueAnnotation,
        current_state: DialogueState,
        information_need: InformationNeed,
    ) -> int:
        """Gets the label for a slot.

        The label is a number in [0,5] which represents the following values:
        0: The slot's value is not mentioned in the user utterance.
        1: The slot's value is set to "dontcare".
        2: The slot's value is requested by the user.
        3: The slot's value is taken from the information need.
        4: The slot's value was previously mentioned and is retrieved from the
          belief state.
        5: The slot's value is randomly chosen.

        Args:
            annotation: Annotation.
            current_state: Current state.
            information_need: Information need.

        Returns:
            Label.
        """
        if annotation.value == "dontcare":
            return 1
        elif annotation.value is None:
            # The value is requested by the user
            return 2
        elif annotation.value == information_need.get_constraint_value(
            annotation.slot
        ) or annotation.value == information_need.requested_slots.get(
            annotation.slot
        ):
            # The value is taken from the information need
            return 3
        elif annotation.value == current_state.belief_state.get(
            annotation.slot
        ):
            # The value was previously mentioned and is
            # retrieved from the belief state
            return 4
        elif (
            annotation.slot
            in [
                information_need.constraints.keys(),
                information_need.requested_slots.keys(),
                current_state.belief_state.keys(),
            ]
        ) and annotation.value not in [
            information_need.get_constraint_value(annotation.slot),
            information_need.requested_slots.get(annotation.slot),
            current_state.belief_state.get(annotation.slot),
        ]:
            # The slot's value is randomly chosen
            return 5
        return 0

    def save_handler(self, path: str) -> None:
        """Saves the feature handler.

        Args:
            path: Path to the output file.
        """
        if not os.path.exists(os.path.dirname(path)):
            logging.info(f"Creating directory: {os.path.dirname(path)}")
            os.makedirs(os.path.dirname(path))

        joblib.dump(self, path)

    @classmethod
    def load_handler(cls, path: str) -> TUSFeatureHandler:
        """Loads a feature handler from a given path.

        Args:
            path: Path to load the feature handler from.

        Raises:
            FileNotFoundError: If the file is not found.

        Returns:
            Feature handler.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File '{path}' not found.")

        return joblib.load(path)
