"""User simulator abstract class."""

from abc import ABC, abstractmethod

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.user import User, UserType
from usersimcrs.core.information_need import generate_random_information_need
from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item_collection import ItemCollection

from usersimcrs.core.information_need import generate_random_information_need
from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item_collection import ItemCollection


class UserSimulator(User, ABC):
    def __init__(
        self,
        id: str,
        domain: SimulationDomain,
        item_collection: ItemCollection,
    ) -> None:
        """Initializes the user simulator."""
        super().__init__(id, UserType.SIMULATOR)
        self._domain = domain
        self._item_collection = item_collection
        self.get_new_information_need()

    def get_new_information_need(self) -> None:
        """Generates a new information need."""
        self.information_need = generate_random_information_need(
            self._domain, self._item_collection
        )

    @abstractmethod
    def _generate_response(self, agent_utterance: Utterance) -> Utterance:
        """Generates response to the agent utterance.

        Args:
            agent_utterance: Agent utterance.

        Raises:
            NotImplementedError: If not implemented in derived class.

        Returns:
            User utterance.
        """
        raise NotImplementedError

    def receive_utterance(self, utterance: Utterance) -> None:
        """Gets called every time there is a new agent utterance.

        Args:
            utterance: Agent utterance.
        """
        response = self._generate_response(utterance)
        if not isinstance(response, AnnotatedUtterance):
            response = AnnotatedUtterance.from_utterance(response)
        self._dialogue_connector.register_user_utterance(response)
