"""Abstract class for user preference modeling.

Currently, there is no difference made between "seen" and "liked" information.
"""

from typing import Dict, List

from dialoguekit.core.annotation import Annotation
from dialoguekit.core.intent import Intent
from dialoguekit.core.ontology import Ontology
from dialoguekit.core.utterance import Utterance


class PreferenceModel:
    """Representation of the user's preferences."""

    def __init__(self, ontology: Ontology) -> None:
        """Initializes the user's preference model.

        Args:
            ontology: An ontology.
        """
        # For each type of entity in the ontology, the user can have
        # preferences, which are stored as entity-preference pairs.
        self.__preferences = {
            class_name: {} for class_name in ontology.get_slot_names()
        }

    # def load_db(self) -> None:
    #     """Loads db/csv file as backend knowledge."""
    #     pass

    def initialize_preferences(self) -> None:
        """Initializes the user's preferences via sampling items."""
        # We assume item-based preferences.
        # We infer preference for classes based on item attributes.
        pass

    def set_preference(self, class_name: str, entity: str, preference: int) -> None:
        """Sets (or updates) preference for a given entity.

        Args:
            class_name: Name of class in the ontology.
            entity: Name of the entity for which the preference is set.
            preference: Preference, represented as an int (negative
                value=dislike, 0=neutral, positive value=like).
        """
        self.__preferences[class_name][entity] = preference

    def get_entity_preference(self, class_name: str, entity: str) -> int:
        """Determines the preference for a given entity.

        Args:
            class_name: Class.
            entity: Entity.

        Return: Preference.
        """
        # TODO: Move this part to CRYSES?
        # If the entity is not in the preference model, we need to infer it.
        if entity not in self.__preferences[class_name]:
            # TODO: infer preference (informed by ontology)
            self.__preferences[class_name][entity] = 0
        return self.__preferences[class_name][entity]

    def get_preference(self, agent_utterance: Utterance) -> int:
        """Determines the preference with regards to an agent recommendation.
        For simplicity, for the time being it's a single number."""
        preference = 0
        for annotation in agent_utterance.get_annotations():
            # Infer preference for the given annotation.
            entity_preference = self.get_entity_preference(annotation.slot, annotation.value)
        # TODO: determine overall preference
        return preference

    def get_next_user_slots(
        self, agent_intent: Intent, agent_slot_values: List[Annotation]
    ) -> Dict:
        """Determines the next user slots via loading from the initialized
        preferences or sampling."""
        # TODO
        pass

