"""Agenda-based user simulator from [Zhang and Balog, KDD'20]."""

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.annotation import Annotation
from dialoguekit.core.domain import Domain
from dialoguekit.core.utterance import Utterance
from dialoguekit.nlg import ConditionalNLG
from dialoguekit.nlu.nlu import NLU

from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.items.ratings import Ratings
from usersimcrs.simulator.agenda_based.interaction_model import InteractionModel
from usersimcrs.simulator.user_simulator import UserSimulator
from usersimcrs.user_modeling.preference_model import PreferenceModel


class AgendaBasedSimulator(UserSimulator):
    def __init__(
        self,
        id: str,
        preference_model: PreferenceModel,
        interaction_model: InteractionModel,
        nlu: NLU,
        nlg: ConditionalNLG,
        domain: Domain,
        item_collection: ItemCollection,
        ratings: Ratings,
    ) -> None:
        """Initializes the agenda-based simulated user.

        Args:
            preference_model: Preference model.
            interaction_model: Interaction model.
            nlu: NLU module performing intent classification and entity linking.
            nlg: NLG module generating textual responses.
            domain: Domain.
            item_collection: Item collection.
            ratings: Historical ratings.
        """
        super().__init__(id=id)
        self._preference_model = preference_model
        self._interaction_model = interaction_model
        self._interaction_model.initialize_agenda()
        self._nlu = nlu
        self._nlg = nlg
        self._domain = domain
        self._item_collection = item_collection
        self._ratings = ratings

    def _generate_response(self, agent_utterance: Utterance) -> Utterance:
        """Generates response to the agent's utterance.

        Args:
            agent_utterance: Agent utterance.

        Return:
            User utterance.
        """
        return self.generate_response(agent_utterance)

    def generate_response(
        self, agent_utterance: Utterance
    ) -> AnnotatedUtterance:
        """Generates response to the agent's utterance.

        Args:
            agent_utterance: Agent utterance.

        Return:
            User utterance.
        """
        # Run agent utterance through NLU.
        agent_annotations = self._nlu.annotate_slot_values(agent_utterance)
        agent_intent = self._nlu.classify_intent(agent_utterance)

        # Test for the agent's stopping intent. Note that this would normally
        # handled by the dialogue connector. However, since intent annotations
        # for the agent's utterance are not available when the response is
        # received by the dialogue connector, an extra check is needed here.
        if agent_intent == self._dialogue_connector._agent.stop_intent:
            self._dialogue_connector.close()
            quit()

        # Response generation (intent and slot-values).
        response_intent = None
        response_slot_values = None

        self._interaction_model.update_agenda(agent_intent)
        response_intent = self._interaction_model.current_intent

        # TODO: Refactor the part below to private methods, once the logic
        # is clear.

        # Agent wants to elicit preferences.
        if self._interaction_model.is_agent_intent_elicit(agent_intent):
            # TODO: Extract the slots from agent response on which preferences
            # are elicited. For now, we just focus on a single slot.
            slot = None

            # Agent is soliciting preferences on a particular slot.
            if slot:
                # TODO: Extract value from agent response.
                value = None
                # Agent is asking about a particular slot-value pair, e.g.,
                # "Do you like action movies?"
                if value:
                    (
                        response_slot,
                        response_value,
                    ) = self._preference_model.get_slot_value_preference(
                        slot, value
                    )
                # Agent is asking about value preferences on a given slot, e.g.,
                # "What movie genre would you prefer?"
                else:
                    (
                        response_slot,
                        response_value,
                    ) = self._preference_model.get_slot_preference(slot)

                if response_slot:
                    response_intent = self._interaction_model.INTENT_DISCLOSE  # type: ignore[attr-defined] # noqa
                    response_slot_values = [
                        Annotation(slot=response_slot, value=response_value)
                    ]
                else:
                    response_intent = self._interaction_model.INTENT_DONT_KNOW  # type: ignore[attr-defined] # noqa

            # Agent is eliciting preferences in general, e.g., "What kind of
            # movies do you like"
            else:
                # TODO: Add method to interaction model that picks both a slot
                # and a preference.
                pass

            # TODO: Check if there are other cases that'd need to be dealt with,
            # based on KDD paper implementation.
            # See: https://github.com/iai-group/UserSimCRS/issues/115

        # Agent is recommending items.
        elif self._interaction_model.is_agent_intent_set_retrieval(
            agent_intent
        ):
            possible_items = (
                self._item_collection.get_items_by_properties(agent_annotations)
                if agent_annotations
                else []
            )
            item_id = possible_items[0].id if possible_items else None

            if item_id is None:
                # The recommended item was not found in the item collection.
                # response_intent = self._interaction_model.INTENT_DONT_KNOW  # type: ignore[attr-defined] # noqa
                response_intent = self._interaction_model.INTENT_NEUTRAL  # type: ignore[attr-defined] # noqa
            else:
                # Determine user preference for the agent's recommendation.
                # First, check if the user has already consumed the item.
                if self._preference_model.is_item_consumed(item_id):
                    # Currently, the user only responds by saying that they
                    # already consumed the item. If there is a follow-up
                    # question by the agent whether they've liked it, that
                    # should end up in the other branch of the fork.
                    response_intent = self._interaction_model.INTENT_ITEM_CONSUMED  # type: ignore[attr-defined] # noqa
                else:
                    # Get a response based on the recommendation. Currently, the
                    # user responds immediately with a like/dislike, but it
                    # could ask questions about the item before deciding (this
                    # should be based on the agenda).
                    preference = self._preference_model.get_item_preference(
                        item_id
                    )
                    if preference > self._preference_model.PREFERENCE_THRESHOLD:
                        response_intent = self._interaction_model.INTENT_LIKE  # type: ignore[attr-defined] # noqa
                    elif (
                        preference
                        < -self._preference_model.PREFERENCE_THRESHOLD
                    ):
                        response_intent = self._interaction_model.INTENT_DISLIKE  # type: ignore[attr-defined] # noqa
                    else:
                        response_intent = self._interaction_model.INTENT_NEUTRAL  # type: ignore[attr-defined] # noqa

        # Generating natural language response through NLG.
        response = self._nlg.generate_utterance_text(
            response_intent, response_slot_values
        )

        return response
