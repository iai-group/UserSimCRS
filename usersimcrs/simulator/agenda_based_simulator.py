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
        super().__init__()
        self._persona = persona
        self._patience = 0
        self._Participant__id = self._persona.id
        self._preference_model = preference_model
        self._preference_model._initialize_item_preferences() #Done in the model init
        self._interaction_model = interaction_model
        self._interaction_model.initialize_agenda() #Done in the model init
        self._nlu = nlu
        self._nlg = nlg
        self._ontology = ontology
        self._item_collection = item_collection
        self._ratings = ratings
        self._satisfaction_model = satisfaction_model
        self._previous_agent_intent = None
        self._previous_user_utterance = None
        self._previous_agent_utterance = None
        print("AGENDA: ", self._interaction_model._agenda)

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
        print("88: agent intent",agent_intent)
        print("89: agent annotations",agent_annotations)

        # Response generation (intent, slot-values and satisfaction).
        response_intent = None
        response_slot_values = None
        response_preference = None
        satisfaction_score = self._satisfaction_model.classify_last_n_dialogue(self._dialogue_manager.dialogue_history,3)
        expected_agent_responses = self._interaction_model._config["expected_responses"][self._interaction_model.current_intent.label]
        print("97: satisfaction score",satisfaction_score)
        print("99: expected responses:", self._interaction_model.current_intent, expected_agent_responses )
        

        # If the is goal met or patience has run out.
        self._patience += 1
        if len(self._dialogue_manager.dialogue_history.utterances)>1:
            if not agent_intent.label in expected_agent_responses:
                if self._patience <= self._persona.max_retries:
                    print("104: repeating: ",self._interaction_model.current_intent)
                    return self._nlg.generate_utterance_text(
                    self._interaction_model.current_intent, response_slot_values, satisfaction=satisfaction_score, force_annotation=True
                )
                self._patience = 0
        self._interaction_model.update_agenda(agent_intent)
        self._previous_agent_intent = agent_intent
        response_intent = self._interaction_model.current_intent
        print("112:response intent",response_intent)
        # Agent wants to elicit preferences.
        if self._interaction_model.is_agent_intent_elicit(agent_intent):
            # Read entities (slots) from preference_model.
            # The choice of slots depends on the agent intent, e.g., REFINE
            # needs replacement items; while EXPAND will need an extra item.
            response_slot_values = self._preference_model.get_next_user_slots(
                agent_intent, agent_annotations
            )
            print("122: user slot values",response_slot_values)
        # Agent is recommending items.
        elif self._interaction_model.is_agent_intent_set_retrieval(
            agent_intent
        ):
            # Determine user preference for the agent's recommendation.
            # Select last annotation since the agent can either give one or two
            # annotations; in both cases, it asks for preference on the last
            # annotation.
            response_preference = self._preference_model.get_item_preference(
                agent_annotations[-1])
            
            if response_intent in self._interaction_model._config["user_preference_intents"]:
                if response_preference[0]:
                    response_intent = random.choice(self._interaction_model._config["user_preference_intents"].get("CONSUMED"))
                else:
                    if response_preference[1] >= 0:
                        response_intent = random.choice(self._interaction_model._config["user_preference_intents"].get("POSITIVE"))
                    else:
                        response_intent = random.choice(self._interaction_model._config["user_preference_intents"].get("NEGATIVE"))
            self._interaction_model._current_intent = response_intent
            print("140: response_preference and intent",response_preference, response_intent)       
                
        # Generating natural language response through NLG.
        response = self._nlg.generate_utterance_text(
            response_intent, response_slot_values, satisfaction=satisfaction_score, force_annotation=True
        )
        print("151: response",response)
        return response