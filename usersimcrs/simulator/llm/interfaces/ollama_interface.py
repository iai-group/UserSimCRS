"""Interface to use a LLM served by Ollama."""

import os

import yaml
from ollama import Client, Options

from dialoguekit.core import Utterance
from dialoguekit.participant import DialogueParticipant
from usersimcrs.simulator.llm.interfaces.llm_interface import LLMInterface
from usersimcrs.simulator.llm.prompt.utterance_generation_prompt import (
    UtteranceGenerationPrompt,
)


class OllamaLLMInterface(LLMInterface):
    def __init__(
        self,
        configuration_path: str,
        default_response: str = None,
    ) -> None:
        """Initializes interface for ollama served LLM.

        Args:
            configuration_path: Path to the configuration file.
            default_response: Default response to be used if the LLM fails to
              generate a response.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            ValueError: If the model or host is not specified in the config.
        """
        super().__init__(default_response)
        if not os.path.exists(configuration_path):
            raise FileNotFoundError(
                f"Configuration file not found: {configuration_path}"
            )

        with open(configuration_path, "r") as f:
            self._llm_configuration = yaml.safe_load(f)

        if "model" not in self._llm_configuration:
            raise ValueError(
                "No model specified in the config, e.g., 'llama2'."
            )
        if "host" not in self._llm_configuration:
            raise ValueError("No host specified in the config.")

        self.client = Client(host=self._llm_configuration.get("host"))
        self.model = self._llm_configuration.get("model")
        self._stream = self._llm_configuration.get("stream", False)
        self._llm_options = Options(
            **self._llm_configuration.get("options", {})
        )

    def generate_response(self, prompt: UtteranceGenerationPrompt) -> Utterance:
        """Generates a user utterance given a prompt.

        Args:
            prompt: Prompt for generating the utterance.

        Returns:
            Utterance.
        """
        response = self.get_llm_response(prompt.prompt_text)
        if response == "":
            response = self.default_response
        response = response.replace("USER: ", "")
        return Utterance(response, participant=DialogueParticipant.USER)

    def get_llm_response(self, prompt: str) -> str:
        """Generates a response given a prompt.

        Args:
            prompt: Prompt for generating the response.

        Returns:
            Response.
        """
        ollama_response = self.client.generate(
            self.model,
            prompt,
            options=self._llm_options,
            stream=self._stream,
        )
        return ollama_response.get("response", "")
