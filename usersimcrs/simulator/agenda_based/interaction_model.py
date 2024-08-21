"""Interaction model.

The interaction model is responsible for defining the allowed transitions
between dialogue acts based on their intents and updating the agenda.
"""

import logging
import os
import random
from collections import defaultdict
from typing import DefaultDict, List, Tuple

import pandas as pd
import yaml
from nltk.stem import WordNetLemmatizer

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.intent import Intent
from dialoguekit.core.slot_value_annotation import SlotValueAnnotation
from dialoguekit.participant import DialogueParticipant
from usersimcrs.core.information_need import InformationNeed
from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.dialogue_management.dialogue_state_tracker import (
    DialogueStateTracker,
)
from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.simulator.agenda_based.agenda import Agenda
from usersimcrs.user_modeling.preference_model import PreferenceModel

_LEMMATIZER = WordNetLemmatizer()

logger = logging.getLogger(__name__)


class InteractionModel:
    """Represents an interaction model."""

    # This set contains the name of the required intents that need to be
    # defined in the configuration file under the field required_intents.
    REQUIRED_INTENTS = {
        "INTENT_START",
        "INTENT_STOP",
        "INTENT_ITEM_CONSUMED",
        "INTENT_LIKE",
        "INTENT_DISLIKE",
        "INTENT_NEUTRAL",
        "INTENT_DISCLOSE",
        "INTENT_INQUIRE",
        "INTENT_YES",
        "INTENT_NO",
        "INTENT_DONT_KNOW",
    }

    def __init__(
        self,
        config_file: str,
        domain: SimulationDomain,
        annotated_conversations: List[Dialogue],
    ) -> None:
        """Initializes the interaction model.

        Args:
            config_file: Path to configuration file.
            domain: Simulation domain.
            annotated_conversations: Annotated conversations.
        """
        # Load interaction model.
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file) as yaml_file:
            self._config = yaml.load(yaml_file, Loader=yaml.FullLoader)

        self._initialize_required_intents()
        self._domain = domain
        (
            self.transition_matrix_single,
            self.transition_matrix_compound,
        ) = self.initialize_transition_matrices(annotated_conversations)

        # Keep track of the dialogue state.
        self.dialogue_state_tracker = DialogueStateTracker()

        # Keep track of the current dialogue acts.
        self._current_dialogue_acts: List[DialogueAct] = []

    def _initialize_required_intents(self) -> None:
        """Initializes required intents.

        Raises:
            RuntimeError: if some required intents are not defined.
        """
        required_intents = self._config.get("required_intents", {})
        if not self.REQUIRED_INTENTS.issubset(required_intents.keys()):
            raise RuntimeError(
                f'The interaction model {self._config.get("name")} needs to '
                "define the following intents under required_intents: "
                f"{self.REQUIRED_INTENTS}"
            )

        for k, v in required_intents.items():
            setattr(self, k, Intent(v))

    def initialize_transition_matrices(
        self, annotated_conversations: List[Dialogue]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Initializes transition matrices from annotated conversations.

        We consider two transition matrices. The first one uses single intents
        as states, while the second one uses compound intents. For example:
        Dialogue acts in utt. 1: [GREETING(), REQUEST(genre=?, year=?)]
        Dialogue acts in utt. 2: [INFORM(genre=action, year=2024)]

        The single intent transition matrix will be:
        GREETING -> INFORM : 1
        REQUEST -> INFORM : 1
        The compound intent transition matrix will be:
        GREETING_REQUEST -> INFORM : 1

        Note that the compound intent may also include single intents, in case
        an utterance has a single dialogue act.

        Args:
            annotated_conversations: Annotated conversations.

        Returns:
            Transition matrices.
        """
        single_intent_distribution: DefaultDict[
            str, DefaultDict[str, int]
        ] = defaultdict(lambda: defaultdict(int))
        compound_intent_distribution: DefaultDict[
            str, DefaultDict[str, int]
        ] = defaultdict(lambda: defaultdict(int))

        agent_user_interactions = self._get_agent_user_interactions(
            annotated_conversations
        )
        for agent_dialogue_acts, user_dialogue_acts in agent_user_interactions:
            compound_agent_intent = self._get_compound_intent_label(
                agent_dialogue_acts
            )
            compound_user_intent = self._get_compound_intent_label(
                user_dialogue_acts
            )
            compound_intent_distribution[compound_agent_intent][
                compound_user_intent
            ] += 1
            for agent_dialogue_act in agent_dialogue_acts:
                agent_intent = agent_dialogue_act.intent.label
                for user_dialogue_act in user_dialogue_acts:
                    user_intent = user_dialogue_act.intent.label
                    single_intent_distribution[agent_intent][user_intent] += 1

        transition_matrix_single = pd.DataFrame.from_dict(
            single_intent_distribution, orient="index"
        ).fillna(0)
        transition_matrix_compound = pd.DataFrame.from_dict(
            compound_intent_distribution, orient="index"
        ).fillna(0)
        # Normalize the transition matrices.
        transition_matrix_single = transition_matrix_single.div(
            transition_matrix_single.sum(axis=1), axis=0
        )
        transition_matrix_compound = transition_matrix_compound.div(
            transition_matrix_compound.sum(axis=1), axis=0
        )

        return transition_matrix_single, transition_matrix_compound

    def _get_agent_user_interactions(
        self, annotated_conversations: List[Dialogue]
    ) -> List[Tuple[List[DialogueAct], List[DialogueAct]]]:
        """Returns agent-user interactions from annotated conversations.

        Args:
            annotated_conversations: Annotated conversations.

        Returns:
            Agent-user interactions.
        """
        agent_user_interactions = []
        for conversation in annotated_conversations:
            agent_dialogue_acts = []
            user_dialogue_acts = []
            for utterance in conversation.utterances:
                assert isinstance(utterance, AnnotatedUtterance), TypeError(
                    "AnnotatedUtterance expected, but found "
                    f"{type(utterance)}"
                )
                if utterance.participant == DialogueParticipant.AGENT:
                    agent_dialogue_acts.extend(utterance.dialogue_acts)
                elif (
                    not agent_dialogue_acts
                    and utterance.participant == DialogueParticipant.USER
                ):
                    # Skip user utterances that come before the agent's first
                    # utterance.
                    continue
                else:
                    user_dialogue_acts.extend(utterance.dialogue_acts)
                if agent_dialogue_acts and user_dialogue_acts:
                    agent_user_interactions.append(
                        (agent_dialogue_acts, user_dialogue_acts)
                    )
                    agent_dialogue_acts = []
                    user_dialogue_acts = []
        return agent_user_interactions

    def initialize_agenda(self, information_need: InformationNeed):
        """Initializes user agenda.

        Args:
            information_need: Information need.
        """
        self.agenda = Agenda(
            information_need,
            self.INTENT_DISCLOSE,  # type: ignore[attr-defined]
            self.INTENT_INQUIRE,  # type: ignore[attr-defined]
            self.INTENT_STOP,  # type: ignore[attr-defined]
            self.INTENT_START,  # type: ignore[attr-defined]
        )

    def is_agent_intent_elicit(self, agent_intent: Intent) -> bool:
        """Checks if the given agent intent is elicitation.

        Args:
            agent_intent: Agent's intent.

        Returns:
            True if it is an elicitation intent.
        """
        return agent_intent.label in self._config["agent_elicit_intents"]

    def is_agent_intent_set_retrieval(self, agent_intent: Intent) -> bool:
        """Checks if the given agent intent is set retrieval.

        Args:
            agent_intent: Agent's intent.

        Returns:
            True if it is a set retrieval intent.
        """
        return agent_intent.label in self._config["agent_set_retrieval"]

    def is_agent_intent_inquire(self, agent_intent: Intent) -> bool:
        """Checks if the given agent intent is inquiry.

        Args:
            agent_intent: Agent's intent.

        Returns:
            True if it is an inquiry intent.
        """
        return agent_intent.label in self._config["agent_inquire_intents"]

    def _is_transition_allowed(
        self, agent_dialogue_acts: List[DialogueAct]
    ) -> bool:
        """Checks if the transition to the next dialogue act is allowed.

        As utterances can have multiple dialogue acts, we consider that if one
        transition out of all possible transitions is allowed, then the
        transition is allowed.

        Args:
            agent_dialogue_acts: Agent's dialogue acts.

        Returns:
            True if the transition is allowed.
        """
        expected_agent_intents = []
        for user_dialogue_act in self._current_dialogue_acts:
            expected_agent_intents.extend(
                self._config["user_intents"]
                .get(user_dialogue_act.intent.label, {})
                .get("expected_agent_intents", [])
            )

        return any(
            agent_dialogue_act.intent.label in expected_agent_intents
            for agent_dialogue_act in agent_dialogue_acts
        )

    def get_next_dialogue_acts(self, n: int = 1) -> List[DialogueAct]:
        """Returns the next n dialogue acts from the stack.

        Args:
            n: Number of dialogue acts to return. Defaults to 1.

        Returns:
            List of dialogue acts.
        """
        dialogue_acts = self.agenda.get_next_dialogue_acts(n)
        self._current_dialogue_acts = dialogue_acts

        return dialogue_acts

    def update_agenda(
        self,
        information_need: InformationNeed,
        preference_model: PreferenceModel,
        item_collection: ItemCollection,
    ) -> None:
        """Updates the agenda based on the last agent dialogue acts and state.

        Each agent dialogue act results in a push operation on the stack. We
        consider four cases: the agent elicits, recommends, inquires, or
        neither. Once the push operations are done, we clean the stack.

        Args:
            information_need: Information need.
        """
        current_state = self.dialogue_state_tracker.get_current_state()
        agent_dialogue_acts = current_state.agent_dialogue_acts[-1]
        user_dialogue_acts = []
        for dialogue_act in agent_dialogue_acts:
            if self.is_agent_intent_elicit(dialogue_act.intent):
                # The agent is eliciting information from the user.
                user_dialogue_acts = (
                    self._generate_elicit_response_dialogue_acts(
                        dialogue_act, information_need, preference_model
                    )
                )
            elif self.is_agent_intent_set_retrieval(dialogue_act.intent):
                # The agent is recommending an item.
                user_dialogue_acts = (
                    self._generate_item_preference_response_dialogue_acts(
                        dialogue_act, preference_model, item_collection
                    )
                )
            elif self.is_agent_intent_inquire(dialogue_act.intent):
                # The agent is inquiring if the user wants to know more about
                # a particular item. The user either inquire a requestable slot
                # or a random slot.
                user_dialogue_acts = (
                    self._generate_inquire_response_dialogue_acts(
                        dialogue_act, information_need
                    )
                )

        if not user_dialogue_acts and not self._is_transition_allowed(
            agent_dialogue_acts
        ):
            # The agent is neither eliciting, inquiring, nor recommending.
            # The next dialogue act in the stack is not allowed by the
            # interaction model, so a new dialogue act is generated based on
            # probability distribution.
            user_dialogue_acts = self._sample_next_user_dialogue_acts(
                information_need, agent_dialogue_acts
            )

        self.agenda.push_dialogue_acts(user_dialogue_acts)
        self.agenda.clean_agenda(information_need)

    def _get_preference_intent(
        self, preference: float, preference_model: PreferenceModel
    ) -> Intent:
        """Returns the preference intent based on the preference value.

        Args:
            preference: Preference value.

        Returns:
            Intent.
        """
        if preference > preference_model.PREFERENCE_THRESHOLD:
            return self.INTENT_LIKE  # type: ignore[attr-defined]

        if preference < -preference_model.PREFERENCE_THRESHOLD:
            return self.INTENT_DISLIKE  # type: ignore[attr-defined]

        return self.INTENT_NEUTRAL  # type: ignore[attr-defined]

    def _generate_elicit_response_dialogue_acts(
        self,
        agent_dialogue_act: DialogueAct,
        information_need: InformationNeed,
        preference_model: PreferenceModel,
    ) -> List[DialogueAct]:
        """Generates dialogue acts for eliciting preferences.

        Args:
            agent_dialogue_act: Agent's dialogue act.
            information_need: Information need.
            preference_model: Preference model.

        Returns:
            List of dialogue acts.
        """
        user_dialogue_acts = []
        elicited_slot_values = [
            (a.slot, a.value) for a in agent_dialogue_act.annotations if a.slot
        ]
        # Considering a compliant user, we assume that all the slots are filled.
        # TODO: Include user model to vary the user's compliance.
        for elicited_slot, elicited_value in elicited_slot_values:
            # During training of the slot annotator, a slot's name and value
            # can be the almost the same, e.g., (GENRE, genres). In that case,
            # value does not represent an entity.
            # TODO: Revise annotator to avoid this behavior.
            # See: https://github.com/iai-group/DialogueKit/issues/234
            elicited_value = (
                None
                if elicited_value
                and _LEMMATIZER.lemmatize(elicited_value).lower()
                == _LEMMATIZER.lemmatize(elicited_slot).lower()
                else elicited_value
            )

            # Agent is asking about a particular slot-value pair, e.g.,
            # "Do you like action movies?"
            if elicited_value:
                preference = preference_model.get_slot_value_preference(
                    elicited_slot, elicited_value
                )
                user_dialogue_acts.append(
                    DialogueAct(
                        self._get_preference_intent(
                            preference, preference_model
                        ),
                        [SlotValueAnnotation(elicited_slot, elicited_value)],
                    )
                )
            else:
                # Agent is asking about value preferences on a given slot, e.g.,
                # "What movie genre would you prefer?" The value is taken either
                # from the information need or the preference model.
                value = None
                if elicited_slot in information_need.constraints:
                    value = information_need.get_constraint_value(elicited_slot)
                else:
                    value, _ = preference_model.get_slot_preference(
                        elicited_slot
                    )

                if value:
                    if isinstance(value, list):
                        annotations = [
                            SlotValueAnnotation(elicited_slot, v) for v in value
                        ]
                    else:
                        annotations = [
                            SlotValueAnnotation(elicited_slot, value)
                        ]
                    user_dialogue_acts.append(
                        DialogueAct(
                            self.INTENT_DISCLOSE,  # type: ignore[attr-defined]
                            annotations,
                        )
                    )
                else:
                    user_dialogue_acts.append(DialogueAct(self.INTENT_DONT_KNOW))  # type: ignore[attr-defined] # noqa

        return user_dialogue_acts

    def _generate_item_preference_response_dialogue_acts(
        self,
        agent_dialogue_act: DialogueAct,
        preference_model: PreferenceModel,
        item_collection: ItemCollection,
    ) -> List[DialogueAct]:
        """Generates dialogue acts for item preference response.

        Args:
            agent_dialogue_act: Agent's dialogue act.
            preference_model: Preference model.
            item_collection: Item collection.

        Returns:
            List of dialogue acts.
        """
        user_dialogue_acts = []
        possible_items = item_collection.get_items_by_properties(
            agent_dialogue_act.annotations
        )
        if not possible_items:
            # The recommended item was not found in the item collection.
            return [DialogueAct(self.INTENT_DONT_KNOW)]  # type: ignore[attr-defined] # noqa

        # Considering a compliant user, we assume that a feedback is given for
        # all recommendations.
        # TODO: Include user model to vary the user's compliance.
        for item in possible_items:
            # Check if the user has already consumed the item.
            if preference_model.is_item_consumed(item.id):
                # Currently, the user only responds by saying that they
                # already consumed the item. If there is a follow-up
                # question by the agent whether they've liked it, that
                # should end up in the other branch of the fork.
                user_dialogue_acts.append(
                    DialogueAct(self.INTENT_ITEM_CONSUMED)  # type: ignore[attr-defined] # noqa
                )

            # Get a response based on the recommendation. Currently, the
            # user responds immediately with a like/dislike, but it
            # could ask questions about the item before deciding. This may be
            # based on a user model.
            preference = preference_model.get_item_preference(item.id)
            user_dialogue_acts.append(
                DialogueAct(
                    self._get_preference_intent(preference, preference_model),
                )
            )

        return user_dialogue_acts

    def _generate_inquire_response_dialogue_acts(
        self,
        agent_dialogue_act: DialogueAct,
        information_need: InformationNeed,
    ) -> List[DialogueAct]:
        """Generates dialogue acts for inquiry response.

        Args:
            agent_dialogue_act: Agent's dialogue act.
            information_need: Information need.

        Returns:
            List of dialogue acts.
        """
        user_dialogue_acts = []
        if agent_dialogue_act.annotations:
            # The agent inquires about a particular slot.
            for slot_value_annotation in agent_dialogue_act.annotations:
                slot = slot_value_annotation.slot
                if slot_value_annotation.value is None:
                    if slot in information_need.get_requestable_slots():
                        user_dialogue_acts.append(
                            DialogueAct(
                                self.INTENT_YES, [SlotValueAnnotation(slot)]  # type: ignore[attr-defined] # noqa
                            )
                        )
                    else:
                        user_dialogue_acts.append(
                            DialogueAct(
                                self.INTENT_NO, [SlotValueAnnotation(slot)]  # type: ignore[attr-defined] # noqa
                            ),
                        )

        if len(user_dialogue_acts) == 0:
            # The agent does not inquire about a particular slot. The user
            # chooses one from the requestable slots or a random slot.
            requestable_slots = information_need.get_requestable_slots()
            if requestable_slots:
                slot = random.choice(requestable_slots)
            else:
                slot = random.choice(self._domain.get_requestable_slots())
            user_dialogue_acts.append(
                DialogueAct(self.INQUIRE, [SlotValueAnnotation(slot)])  # type: ignore[attr-defined] # noqa
            )

        return user_dialogue_acts

    def _sample_next_user_dialogue_acts(
        self,
        information_need: InformationNeed,
        agent_dialogue_acts: List[DialogueAct],
    ) -> List[DialogueAct]:
        """Samples next user dialogue acts based on a probability distribution.

        Args:
            information_need: Information need.
            agent_dialogue_acts: Agent's dialogue acts.

        Returns:
            List of dialogue acts.
        """
        sampled_user_intents = self._sample_user_intents(agent_dialogue_acts)

        # Generate dialogue acts based on the sampled intents, the annotations
        # are generated based on the information need and belief state. Note
        # that only simple cases (disclose and inquire) are considered here.
        user_dialogue_acts = []
        current_belief_state = (
            self.dialogue_state_tracker.get_current_state().belief_state
        )
        for sampled_intent in sampled_user_intents:
            if sampled_intent == self.INTENT_DISCLOSE:  # type: ignore[attr-defined] # noqa
                # Check if there is a slot from the information need that is
                # not fulfilled in the belief state, else choose a random slot
                # value from the constraints.
                slot = None
                value = None
                for belief_state_slot in current_belief_state.keys():
                    if belief_state_slot not in information_need.constraints:
                        slot = belief_state_slot
                        value = information_need.get_constraint_value(slot)
                        break

                if slot is None:
                    slot, value = random.choice(
                        list(information_need.constraints.items())
                    )

                user_dialogue_acts.append(
                    DialogueAct(
                        sampled_intent, [SlotValueAnnotation(slot, value)]
                    )
                )
            elif sampled_intent == self.INTENT_INQUIRE:  # type: ignore[attr-defined] # noqa
                slot = random.choice(information_need.get_requestable_slots())
                if not slot:
                    slot = random.choice(self._domain.get_requestable_slots())
                user_dialogue_acts.append(
                    DialogueAct(sampled_intent, [SlotValueAnnotation(slot)])
                )
            else:
                user_dialogue_acts.append(DialogueAct(sampled_intent))

        return user_dialogue_acts

    def _get_compound_intent_label(
        self, dialogue_acts: List[DialogueAct]
    ) -> str:
        """Returns the compound intent label.

        The compound intent label is formed by concatenating alphabetically
        sorted single intent labels with underscore as separator.

        Args:
            dialogue_acts: Dialogue acts.

        Returns:
            Compound intent label.
        """
        return "_".join(sorted({da.intent.label for da in dialogue_acts}))

    def _sample_user_intents(self, agent_dialogue_acts: List[DialogueAct]):
        """Samples user intents based on the agent's dialogue acts.

        Checks if the agent's dialogue acts are in the compound transition
        matrix. If not, we consider the single transition matrix.

        Args:
            agent_dialogue_acts: Agent's dialogue acts.

        Returns:
            List of sampled user intents.
        """
        compound_agent_intent = self._get_compound_intent_label(
            agent_dialogue_acts
        )
        sampled_user_intents = []

        if compound_agent_intent in self.transition_matrix_compound.index:
            # Sample from the row of the compound transition matrix.
            user_intents = self.transition_matrix_compound.loc[
                compound_agent_intent
            ]
            sampled_user_intents.extend(
                [
                    Intent(intent)
                    for intent in user_intents.sample(n=1, weights=user_intents)
                    .index[0]
                    .split("_")
                ]
            )
        else:
            # For each agent's dialogue act, we sample from the row of the
            # single intent transition matrix.
            for agent_dialogue_act in agent_dialogue_acts:
                try:
                    user_intents = self.transition_matrix_single.loc[
                        agent_dialogue_act.intent.label
                    ]
                    sampled_user_intents.append(
                        Intent(
                            user_intents.sample(
                                n=1, weights=user_intents
                            ).index[0]
                        )
                    )
                except KeyError:
                    logger.warning(
                        f"Transition matrix does not contain agent intent: "
                        f"{agent_dialogue_act.intent.label}"
                    )
                    continue
        return sampled_user_intents
