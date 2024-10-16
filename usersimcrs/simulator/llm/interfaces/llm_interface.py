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
    def generate_natural_response(
        self, prompt: UtteranceGenerationPrompt
    ) -> Utterance:
        """Generates an utterance given a prompt.

        Args:
            prompt: Prompt for generating the utterance.

        Raises:
            NotImplementedError: If the method is not implemented in subclass.

        Returns:
            Utterance in natural language.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_llm_api_response(self, prompt: str, **kwargs) -> str:
        """Gets the raw response from the LLM API.

        This method should be used to interact directly with the LLM API, i.e.,
        for everything that is not related to the generation of an utterance.

        Args:
            prompt: Prompt for the LLM.
            **kwargs: Additional arguments to be passed to the API call.

        Raises:
            NotImplementedError: If the method is not implemented in subclass.

        Returns:
            Response from the LLM API without any post-processing.
        """
        raise NotImplementedError()
