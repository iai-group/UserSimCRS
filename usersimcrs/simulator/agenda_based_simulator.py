"""Agenda-based user simulator from [Zhang and Balog, KDD'20]."""
import random

from dialoguekit.core.ontology import Ontology
from dialoguekit.core.recsys.item_collection import ItemCollection
from dialoguekit.core.recsys.ratings import Ratings
from dialoguekit.nlg.nlg import NLG
from dialoguekit.nlu.nlu import NLU
from dialoguekit.core.utterance import Utterance
from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.nlu.models.satisfaction_classifier import SatisfactionClassifier
from dialoguekit.core.intent import Intent
from dialoguekit.user.user import UserType

from usersimcrs.simulator.preference_model import PreferenceModel
from usersimcrs.simulator.user_simulator import UserSimulator
from usersimcrs.utils.persona_generator import Persona
from usersimcrs.simulator.interaction_model import InteractionModel


class AgendaBasedSimulator(UserSimulator):
    """Agenda based user simulator."""

    def __init__(
        self,
        id: str,
        preference_model: PreferenceModel,
        interaction_model: InteractionModel,
        nlu: NLU,
        nlg: NLG,
        ontology: Ontology,
        item_collection: ItemCollection,
        ratings: Ratings,
        persona: Persona,
        satisfaction_model: SatisfactionClassifier
    ) -> None:
        """ "
        Initializes the agenda-based simulated user.

        Args:
            preference_model: Preference model.
            interaction_model: Interaction model.
            nlu: NLU module performing intent classification and entity linking.
            nlg: NLG module generating textual responses.
            ontology: Ontology.
            item_collection: Item collection.
            ratings: Historical ratings.
        """
        super().__init__(id=persona.id)
        self._persona = persona
        self._patience = 0
        self._Participant__id = self._persona.id
        self._preference_model = preference_model
        # self._preference_model._initialize_item_preferences() #Done in the model init
        self._interaction_model = interaction_model
        # self._interaction_model.initialize_agenda() #Done in the model init
        self._nlu = nlu
        self._nlg = nlg
        self._ontology = ontology
        self._item_collection = item_collection
        self._ratings = ratings
        self._satisfaction_model = satisfaction_model
        self._previous_agent_intent = None
        self._previous_user_utterance = None
        self._previous_agent_utterance = None

    def _generate_response(self, agent_utterance: Utterance) -> Utterance:
        """Generate response to the agent utterance.

        Args:
            agent_utterance: Agent utterance.

        Return:
            User utterance.
        """
        return self.generate_response(agent_utterance)

    def generate_response(self, agent_utterance: Utterance) -> Utterance:
        """Generate response to the agent utterance.

        Args:
            agent_utterance: Agent utterance.

        Return:
            User utterance.
        """
        # Run agent utterance through NLU.
        agent_annotations = self._nlu.annotate_slot_values(agent_utterance)
        agent_intent = self._nlu.get_intent(agent_utterance)

        # Response generation (intent, slot-values and satisfaction).
        response_intent = None
        response_slot_values = None
        response_preference = None
        satisfaction_score = self._satisfaction_model.classify_last_n_dialogue(self._dialogue_manager.dialogue_history,3)
        expected_agent_responses = self._interaction_model._config["expected_responses"][self._interaction_model.current_intent.label]

        # If the is goal met or patience has run out.
        self._patience += 1
        if len(self._dialogue_manager.dialogue_history.utterances)>1:
            if not agent_intent.label in expected_agent_responses:
                if self._patience <= self._persona._max_retries:
                    premature_response = self._nlg.generate_utterance_text(
                    self._interaction_model.current_intent, response_slot_values, satisfaction=satisfaction_score, force_annotation=True
                )
                    if premature_response is False:
                        premature_response = self._nlg.generate_utterance_text(
                    self._interaction_model.current_intent, response_slot_values, satisfaction=satisfaction_score, force_annotation=False
                )
                    return premature_response
        self._patience = 0
        self._interaction_model.update_agenda(agent_intent)
        self._previous_agent_intent = agent_intent
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
            # Select last annotation since the agent can either give one or two
            # annotations; in both cases, it asks for preference on the last
            # annotation.
            if len(agent_annotations)>0:
                response_preference = self._preference_model.get_item_preference(
                    agent_annotations[-1].value)
                if response_intent.label in self._interaction_model._config["user_preference_intents"]:
                    if response_preference[0]:
                        response_intent = Intent(random.choice(self._interaction_model._config["user_preference_intents"].get(response_intent.label).get("CONSUMED")))
                    else:
                        if response_preference[1] >= 0:
                            response_intent = Intent(random.choice(self._interaction_model._config["user_preference_intents"].get(response_intent.label).get("POSITIVE")))
                        else:
                            response_intent = Intent(random.choice(self._interaction_model._config["user_preference_intents"].get(response_intent.label).get("NEGATIVE")))
                self._interaction_model._current_intent = response_intent
        if response_intent.label in self._interaction_model._config["user_remove_preference_intents"]:
            response_slot_values = self._preference_model.revise_random_preference()
        # Generating natural language response through NLG.
        response = self._nlg.generate_utterance_text(
            response_intent, response_slot_values, satisfaction=satisfaction_score, force_annotation=True
        )
        if response is False:
            for i in range(3):
                response = self._nlg.generate_utterance_text(
                response_intent, response_slot_values, satisfaction=satisfaction_score, force_annotation=False
                )
                if response:
                    break
        if response is False:
            response = AnnotatedUtterance("Can you repeat?",response_intent,annotations=None,satisfaction=satisfaction_score)
        self._preference_model._set_session_preference(response.get_annotations())
        return response