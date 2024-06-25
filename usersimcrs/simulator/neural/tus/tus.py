"""Transformer-based User Simulator (TUS)

Reference: Domain-independent User Simulation with Transformers for
Task-oriented Dialogue Systems, Lin et al., 2021.
See: https://arxiv.org/abs/2106.08838

Implementation is adapted from the description in the paper and the original
implementation by the authors:
https://gitlab.cs.uni-duesseldorf.de/general/dsml/tus_public
"""

import logging
import random
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List

import torch
from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.annotation import Annotation
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.utterance import Utterance
from dialoguekit.nlu.nlu import NLU
from dialoguekit.participant import DialogueParticipant

from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.dialogue_management.dialogue_state_tracker import (
    DialogueStateTracker,
)
from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.simulator.neural.core.feature_handler import (
    FeatureMask,
    FeatureVector,
)
from usersimcrs.simulator.neural.core.transformer import (
    TransformerEncoderModel,
)
from usersimcrs.simulator.neural.tus.tus_feature_handler import (
    TUSFeatureHandler,
)
from usersimcrs.simulator.user_simulator import UserSimulator

logger = logging.getLogger(__name__)


class TUS(UserSimulator):
    def __init__(
        self,
        id: str,
        domain: SimulationDomain,
        item_collection: ItemCollection,
        nlu: NLU,
        feature_handler: TUSFeatureHandler,
        dialogue_state_tracker: DialogueStateTracker,
        network_config: Dict[str, Any],
    ) -> None:
        """Initializes the Transformer-based User Simulator (TUS).

        Args:
            id: Simulator ID.
            domain: Domain knowledge.
            item_collection: Collection of items.
            nlu: NLU module.
            feature_handler: Feature handler.
            dialogue_state_tracker: Dialogue state tracker.
            network_config: Network configuration.
        """
        super().__init__(id=id, domain=domain, item_collection=item_collection)
        self._nlu = nlu
        self._feature_handler = feature_handler
        self._user_policy_network = TransformerEncoderModel(**network_config)
        self._dialogue_state_tracker = dialogue_state_tracker
        self._last_user_actions: DefaultDict[str, torch.Tensor] = defaultdict(
            lambda: torch.tensor([])
        )
        self._last_turn_input: torch.Tensor = None

    def initialize(self) -> None:
        """Initializes the user simulator."""
        self._dialogue_state_tracker.reset_state()
        self._last_user_actions.clear()
        self._last_turn_input = None

    def _generate_response(self, agent_utterance: Utterance) -> Utterance:
        """Generates response to the agent utterance.

        Args:
            agent_utterance: Agent utterance.

        Returns:
            User utterance.
        """
        previous_state = self._dialogue_state_tracker.get_current_state()
        # 1. Perform NLU on the agent utterance, i.e., extract dialogue acts or
        # intent and annotations.
        annotated_agent_utterance = self._annotate_agent_utterance(
            agent_utterance
        )

        # 2. Update dialogue state based on the agent utterance.
        self._dialogue_state_tracker.update_state(
            dialogue_acts=annotated_agent_utterance.dialogue_acts,
            participant=DialogueParticipant.AGENT,
        )

        # 3. Extract features for the current turn.
        turn_feature, mask = self._feature_handler.build_input_vector(
            agent_dialogue_acts=annotated_agent_utterance.dialogue_acts,
            previous_state=previous_state,
            state=self._dialogue_state_tracker.get_current_state(),
            information_need=self.information_need,
            user_action_vectors=self._last_user_actions,
        )

        # 5. Predict user dialogue acts based on the features.
        user_dialogue_acts = self.predict_user_dialogue_acts(
            turn_feature, mask, self._feature_handler.action_slots
        )

        # 6. Generate user utterance based on the predicted actions.
        # For now, we only consider the first predicted dialogue act due to
        # constraints related to supported NLG in DialogueKit.
        response_intent = user_dialogue_acts[0].intent
        response_annotations = user_dialogue_acts[0].annotations
        response = self._nlg.generate_utterance_text(
            intent=response_intent, annotations=response_annotations
        )
        response.participant = DialogueParticipant.USER

        # 7. Update dialogue state based on the user utterance.
        self._dialogue_state_tracker.update_state(
            dialogue_acts=user_dialogue_acts,
            participant=DialogueParticipant.USER,
        )

        return response

    def _annotate_agent_utterance(
        self, agent_utterance: Utterance
    ) -> AnnotatedUtterance:
        """Annotates the agent utterance.

        As of now, DialogueKit does not support NLU that can annotate dialogue
        acts. So, only one dialogue act is created with the intent and
        annotations extracted from the agent utterance.

        Args:
            agent_utterance: Agent utterance.

        Returns:
            Annotated utterance.
        """
        agent_intent = self._nlu.classify_intent(agent_utterance)
        agent_annotations = self._nlu.annotate_slot_values(agent_utterance)
        dialogue_acts = [
            DialogueAct(intent=agent_intent, annotations=agent_annotations)
        ]
        utt = AnnotatedUtterance(
            text=agent_utterance.text,
            participant=DialogueParticipant.AGENT,
            dialogue_acts=dialogue_acts,
        )
        return utt

    def predict_user_dialogue_acts(
        self,
        features: List[FeatureVector],
        mask: FeatureMask,
        action_slots: List[str],
    ) -> List[DialogueAct]:
        """Predicts user dialogue acts based on the features.

        Args:
            features: Feature vector.
            mask: Mask vector.
            action_slots: Action slots used to predict the user action per slot.

        Returns:
            Predicted user dialogue acts.
        """
        output = self._user_policy_network(features, mask)
        # fmt: off
        output = output[
            :, 1 : self._feature_handler.max_turn_feature_length + 1, :  # noqa: E203, E501
        ]
        # fmt: on

        slot_outputs: Dict[str, int] = {}
        for index, slot_name in enumerate(action_slots):
            o = int(torch.argmax(output[0, index + 1, :]).item())
            assert o in range(6), f"Invalid output: {o}"
            slot_outputs[slot_name] = o
            # One-hot encoding of user action for the slot
            o_i = torch.zeros(6)
            o_i[o] = 1
            self._last_user_actions[slot_name] = o_i

        user_dialogue_acts = self._parse_policy_output(
            action_slots, slot_outputs
        )
        return user_dialogue_acts

    def _parse_policy_output(
        self, action_slots: List[str], slot_outputs: Dict[str, int]
    ) -> List[DialogueAct]:
        """Parses the policy output to dialogue acts.

        Args:
            action_slots: Action slots.
            slot_outputs: Output per slot.

        Returns:
            Dialogue acts.
        """
        belief_state = (
            self._dialogue_state_tracker.get_current_state().belief_state
        )
        dialogue_acts = []

        for slot in action_slots:
            o = slot_outputs[slot]
            dialogue_act = DialogueAct()

            # Default intent is "inform"
            dialogue_act.intent = "inform"

            # Determine the value of the slot
            if o == 1:
                # The slot's value is requested by the user
                dialogue_act.intent = "request"
                dialogue_act.annotations.append(Annotation(slot))
            elif o == 2:
                # The slot's value is set to "dontcare"
                dialogue_act.annotations.append(Annotation(slot, "dontcare"))
            elif o == 3:
                # The slot's value is taken from the information need
                if slot in self.information_need.constraints.keys():
                    dialogue_act.annotations.append(
                        Annotation(
                            slot, self.information_need.constraints[slot]
                        )
                    )
            elif o == 4:
                # The slot's value was previously mentioned and is retrieved
                # from the belief state
                if slot in belief_state.keys():
                    dialogue_act.annotations.append(
                        Annotation(slot, belief_state[slot])
                    )
            elif o == 5:
                # The slot's value in the information need is randomly modified
                value = random.choice(
                    list(
                        self._item_collection.get_possible_property_values(slot)
                    )
                )
                self.information_need.constraints[slot] = value
                dialogue_act.annotations.append(Annotation(slot, value))
            else:
                logger.warning(f"{slot} is not mentioned in this turn.")
                continue

            dialogue_acts.append(dialogue_act)

        return dialogue_acts
