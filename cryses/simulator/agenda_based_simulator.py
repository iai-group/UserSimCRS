"""Agenda-based user simulator from [Zhang and Balog, KDD'20]."""

from dialoguekit.nlg.nlg import NLG
from dialoguekit.nlu.nlu import NLU
from dialoguekit.user.preference_model import PreferenceModel
from cryses.simulator.user_simulator import UserSimulator
from dialoguekit.core.utterance import Utterance

from cryses.simulator.interaction_model import InteractionModel


class AgendaBasedSimulator(UserSimulator):
    """Agenda based user simulator."""

    def __init__(
        self,
        preference_model: PreferenceModel,
        interaction_model: InteractionModel,
        nlu: NLU,
        nlg: NLG,
    ) -> None:
        """ "
        Initializes the agenda-based simulated user.

        Args:
            preference_model: Preference model.
            interaction_model: Interaction model.
            nlu: NLU module performing intent classification and entity linking.
            nlg: NLG module generating textual responses.
        """
        super().__init__()
        self.__preference_model = preference_model
        self.__preference_model.initialize_preference()
        self.__interaction_model = interaction_model
        self.__interaction_model.initialize_agenda()
        self.__nlu = nlu
        self.__nlg = nlg

    def _generate_response(self, agent_utterance: Utterance) -> Utterance:
        """Generate response to the agent utterance.

        Args:
            agent_utterance: Agent utterance.

        Return:
            User utterance.
        """
        pass

    def generate_response(self, utterance: Utterance) -> Utterance:
        """Generate response to the agent utterance.

        Args:
            utterance: Agent utterance.

        Return:
            User utterance.
        """
        # Run agent utterance through NLU.
        agent_intent = self.__nlu.get_intent()
        agent_slot_values = self.__nlu.get_slot_values()

        # Response generation (intent and slot-values).
        response_intent = None
        response_slot_values = None

        self.__interaction_model.update_agenda(agent_intent)
        response_intent = self.__interaction_model.current_intent

        # Agent wants to elicit preferences.
        if self.__interaction_model.is_agent_intent_elicit(agent_intent):
            # read slots from preference_model (updates slot_values);
            response_slot_values = self.__preference_model.next_user_slots(agent_intent, agent_slot_values)
        # Agent is recommending items.
        elif self.__interaction_model.is_agent_intent_set_retrieval(agent_intent):
            # Determine response/rating based on entities and preference_model (rates or likes)
            ratings = self.__preference_model.update_agent_slots(agent_slot_values)
            # Update the user preferences
            self.__preference_model.update_preferences(agent_slot_values, ratings)

        # Generating natural language response through NLG.
        response_text = self.__nlg.generate_utterance_text(response_intent, response_slot_values, ratings)

        return Utterance(response_text, response_intent)
