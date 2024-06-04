"""Interface for the LLM model."""

from abc import ABC, abstractmethod

from dialoguekit.core import Utterance

from usersimcrs.simulator.llm.prompt import Prompt


class LLMInterface(ABC):
    def __init__(self, default_response: str = None) -> None:
        """Initializes the LLM interface.

        Args:
            default_response: Default response to be used if the LLM fails to
              generate a response.
        """
        self.default_response = default_response

    @abstractmethod
    def generate_response(self, prompt: Prompt) -> Utterance:
        """Generates an utterance given a prompt.

        Args:
            prompt: Prompt for generating the utterance.

        Raises:
            NotImplementedError: If the method is not implemented in subclass.

        Returns:
            Utterance.
        """
        raise NotImplementedError()
