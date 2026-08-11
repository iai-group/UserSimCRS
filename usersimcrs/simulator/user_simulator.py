"""User simulator abstract class."""

from abc import ABC, abstractmethod

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.user import User, UserType
from usersimcrs.information_need_management.information_need_tracker import (
    InformationNeedTracker,
)
from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item_collection import ItemCollection


class UserSimulator(User, ABC):
    def __init__(
        self,
        id: str,
        domain: SimulationDomain,
        item_collection: ItemCollection,
        information_need_tracker: InformationNeedTracker,
    ) -> None:
        """Initializes the user simulator.

        Args:
            id: Simulator ID.
            domain: Domain.
            item_collection: Item collection.
            information_need_tracker: Tracker for the information need.
        """
        super().__init__(id, UserType.SIMULATOR)
        self._domain = domain
        self._item_collection = item_collection
        self.information_need_tracker = information_need_tracker

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
