"""User simulator abstract class."""

from abc import ABC, abstractmethod

from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.user import User, UserType


class UserSimulator(User, ABC):
    def __init__(
        self,
        id: str,
    ) -> None:
        """Initializes the user simulator."""
        super().__init__(id, UserType.SIMULATOR)

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
        self._dialogue_connector.register_user_utterance(response)
