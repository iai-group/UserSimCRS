"""User simulator abstract class."""

from abc import ABC, abstractmethod

from dialoguekit.user.user import User, UserType
from dialoguekit.core.utterance import Utterance


class UserSimulator(User, ABC):
    def __init__(
        self,
    ) -> None:
        """"Initializes the user simulator."""
        self._user_type = UserType.SIMULATOR

    @abstractmethod
    def _generate_response(self, agent_utterance: Utterance) -> Utterance:
        """Generate response to the agent utterance.

        Args:
            agent_utterance: Agent utterance.

        Return:
            User utterance.
        """
        pass

    def receive_utterance(self, agent_utterance: Utterance) -> None:
        """This method is called each time there is a new agent utterance.

        Args:
            agent_utterance: Agent utterance.
        """
        self._dialogue_manager.register_user_utterance(
            self._generate_response(agent_utterance)
        )
