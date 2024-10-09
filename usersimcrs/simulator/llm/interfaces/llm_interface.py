"""Interface for the LLM model."""

from abc import ABC, abstractmethod

from dialoguekit.core import Utterance
from usersimcrs.simulator.llm.prompt.utterance_generation_prompt import (
    UtteranceGenerationPrompt,
)


class LLMInterface(ABC):
    def __init__(self, default_response: str = None) -> None:
        """Initializes the LLM interface.

        Args:
            default_response: Default response to be used if the LLM fails to
              generate a response.
        """
        self.default_response = default_response

    @abstractmethod
    def generate_response(self, prompt: UtteranceGenerationPrompt) -> Utterance:
        """Generates an utterance given a prompt.

        Args:
            prompt: Prompt for generating the utterance.

        Raises:
            NotImplementedError: If the method is not implemented in subclass.

        Returns:
            Utterance.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_llm_response(self, prompt: str, **kwargs) -> str:
        """Generates a response given a prompt.

        Args:
            prompt: Prompt for generating the response.

        Raises:
            NotImplementedError: If the method is not implemented in subclass.

        Returns:
            Response.
        """
        raise NotImplementedError()
