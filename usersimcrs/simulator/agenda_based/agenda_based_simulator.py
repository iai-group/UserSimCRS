"""Agenda-based user simulator from [Zhang and Balog, KDD'20]."""

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.utterance import Utterance
from dialoguekit.nlg import ConditionalNLG
from dialoguekit.nlu import NLU
from dialoguekit.participant.participant import DialogueParticipant
from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.dialogue_management.dialogue_state_tracker import (
    DialogueStateTracker,
)
from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.items.ratings import Ratings
from usersimcrs.simulator.agenda_based.interaction_model import (
    InteractionModel,
)
from usersimcrs.simulator.user_simulator import UserSimulator
from usersimcrs.user_modeling.preference_model import PreferenceModel


class AgendaBasedSimulator(UserSimulator):
    def __init__(
        self,
        id: str,
        domain: SimulationDomain,
        item_collection: ItemCollection,
        preference_model: PreferenceModel,
        interaction_model: InteractionModel,
        nlu: NLU,
        nlg: ConditionalNLG,
        ratings: Ratings,
    ) -> None:
        """Initializes the agenda-based simulated user.

        Args:
            id: Simulator ID.
            domain: Domain.
            item_collection: Item collection.
            preference_model: Preference model.
            interaction_model: Interaction model.
            nlu: NLU module performing dialogue acts extraction.
            nlg: NLG module generating textual responses.
            ratings: Historical ratings.
        """
        super().__init__(id=id, domain=domain, item_collection=item_collection)
        self._preference_model = preference_model
        self._interaction_model = interaction_model
        self._interaction_model.initialize_agenda(self.information_need)
        self._nlu = nlu
        self._nlg = nlg
        self._dialogue_state_tracker = DialogueStateTracker()
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
        agent_dialogue_acts = self._nlu.extract_dialogue_acts(agent_utterance)

        self._interaction_model.dialogue_state_tracker.update_state(
            agent_dialogue_acts, DialogueParticipant.AGENT
        )

        # Test for the agent's stopping intent. Note that this would normally
        # handled by the dialogue connector. However, since intent annotations
        # for the agent's utterance are not available when the response is
        # received by the dialogue connector, an extra check is needed here.
        if self._dialogue_connector._agent.stop_intent in [
            da.intent for da in agent_dialogue_acts
        ]:
            self._dialogue_connector.close()
            quit()

        # Update agenda based on the agent's dialogue acts.
        self._interaction_model.update_agenda(
            self.information_need,
            self._preference_model,
            self._item_collection,
        )
        # Get next user dialogue acts based on the current agenda.
        response_dialogue_acts = (
            self._interaction_model.get_next_dialogue_acts()
        )

        # Generating natural language response through NLG.
        response = self._nlg.generate_utterance_text(response_dialogue_acts)

        response.participant = DialogueParticipant.USER

        # Update dialogue state.
        self._interaction_model.dialogue_state_tracker.update_state(
            response_dialogue_acts, DialogueParticipant.USER
        )

        return response
