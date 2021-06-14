"""Agenda-based user simulator from [Zhang and Balog, KDD'20]."""

from dialoguekit.nlg.nlg import NLG
from dialoguekit.nlu.nlu import NLU
from dialoguekit.user.preference_model import PreferenceModel
from cryses.simulator.user_simulator import UserSimulator
from dialoguekit.utterance.utterance import Utterance

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
        entities = self.__nlu.get_entities()
        intent = self.__nlu.get_intent()

        # Response generation (intent and slot-values).
        response_intent = None
        slot_values = None
        # TODO(Shuo): add pseudo code
        # self.__interaction_model.update_agenda()
        # response_intent = self.__interaction_model.current_intent
        #
        # if intent lies in query formulation (agent expects preference eliciation):
        #   read slots from preference_model (updates slot_values);
        # elif intent lies in set retrieval (agent is recommending items):
        #   determins rates based on entities and preference_model (rates or likes)


        # Generating natural language response through NLG.
        response_text = self.__nlg.generate_utterance_text(response_intent, slot_values)

        return Utterance(response_text, response_intent)
