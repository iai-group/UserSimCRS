"""Agenda-based user simulator from [Zhang and Balog, KDD'20]."""

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
        # self._preference_model.initialize_preference()
        self._interaction_model = interaction_model
        self._interaction_model.initialize_agenda()
        self._nlu = nlu
        self._nlg = nlg
        self._domain = domain
        self._item_collection = item_collection
        self._ratings = ratings

    def _generate_response(self, agent_utterance: Utterance) -> Utterance:
        """Generate response to the agent utterance.

        Args:
            agent_utterance: Agent utterance.

        Return:
            User utterance.
        """
        self.generate_response(agent_utterance)

    def generate_response(self, agent_utterance: Utterance) -> Utterance:
        """Generate response to the agent utterance.

        Args:
            agent_utterance: Agent utterance.

        Return:
            User utterance.
        """
        # Run agent utterance through NLU.
        self._nlu.annotate_slot_values(agent_utterance)
        agent_intent = self._nlu.get_intent(agent_utterance)
        agent_annotations = agent_utterance.get_annotations()

        # Response generation (intent and slot-values).
        response_intent = None
        response_slot_values = None
        response_preference = None

        self._interaction_model.update_agenda(agent_intent)
        response_intent = self._interaction_model.current_intent

        # Agent wants to elicit preferences.
        if self._interaction_model.is_agent_intent_elicit(agent_intent):
            # Read entities (slots) from preference_model.
            # The choice of slots depends on the agent intent, e.g., REFINE
            # needs replacement items; while EXPAND will need an extra item.
            response_slot_values = self._preference_model.get_next_user_slots(
                agent_intent, agent_annotations
            )
        # Agent is recommending items.
        elif self._interaction_model.is_agent_intent_set_retrieval(
            agent_intent
        ):
            # Determine user preference for the agent's recommendation.
            # TODO: Replace get_preference with appropriate method.
            # See: https://github.com/iai-group/UserSimCRS/issues/88
            response_preference = self._preference_model.get_preference(
                agent_utterance
            )

        # Generating natural language response through NLG.
        response_text = self._nlg.generate_utterance_text(
            response_intent, response_slot_values, response_preference
        )

        return Utterance(response_text, response_intent)
