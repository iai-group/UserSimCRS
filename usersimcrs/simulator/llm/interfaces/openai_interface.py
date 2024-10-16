"""Interface to use a LLM served by OpenAI."""

import os
import re
from typing import Dict, List

import yaml
from openai import OpenAI

from dialoguekit.core import Utterance
from dialoguekit.participant.participant import DialogueParticipant
from usersimcrs.simulator.llm.interfaces.llm_interface import LLMInterface
from usersimcrs.simulator.llm.prompt.utterance_generation_prompt import (
    UtteranceGenerationPrompt,
)


class OpenAILLMInterface(LLMInterface):
    def __init__(
        self,
        configuration_path: str,
        use_chat_api: bool = False,
        default_response: str = None,
    ) -> None:
        """Initializes interface for OpenAI served LLM.

        Args:
            configuration_path: Path to the configuration file.
            use_chat_api: Whether to use the chat or completion API. Defaults to
              False (i.e., completion API).
            default_response: Default response to be used if the LLM fails to
              generate a response.

        Raises:
            FileNotFoundError: If the configuration file is not found.
        """
        super().__init__(default_response)

        if not os.path.exists(configuration_path):
            raise FileNotFoundError(
                f"Configuration file not found: {configuration_path}"
            )

        with open(configuration_path, "r") as f:
            self._llm_configuration = yaml.safe_load(f)

        if "api_key" not in self._llm_configuration:
            raise ValueError(
                "No API key specified in the config, see how to get one at "
                "https://platform.openai.com/docs/quickstart/account-setup"
            )

        if "model" not in self._llm_configuration:
            raise ValueError(
                "No model specified in the config, see supported models at "
                "https://platform.openai.com/docs/models"
            )

        self.model = self._llm_configuration.get("model")
        self._llm_options = self._llm_configuration.get("options", {})

        self.client = OpenAI(api_key=self._llm_configuration.get("api_key"))
        self.use_chat_api = use_chat_api

    def generate_natural_response(
        self, prompt: UtteranceGenerationPrompt
    ) -> Utterance:
        """Generates a user utterance given a prompt.

        Args:
            prompt: Prompt for generating the utterance.

        Returns:
            Utterance in natural language.
        """
        response = self.get_llm_api_response(
            prompt.prompt_text, initial_prompt=prompt.build_new_prompt()
        )
        response = response.replace("USER: ", "")
        return Utterance(response, DialogueParticipant.USER)

    def _parse_prompt_context(
        self, prompt_context: str
    ) -> List[Dict[str, str]]:
        """Parses the prompt context to a list of messages.

        Args:
            prompt_context: Prompt context.
        """
        messages = []
        utterances = prompt_context.split("\n")
        role_pattern = re.compile(r"^\[(USER|ASSISTANT)\]: (.+)$")
        for utterance in utterances:
            match = role_pattern.match(utterance)
            if match:
                role = match.group(1)
                text = match.group(2)
                messages.append({"role": role.lower(), "content": text})
        return messages

    def get_llm_api_response(
        self, prompt: str, initial_prompt: str = None
    ) -> str:
        """Gets the raw response from the LLM API.

        This method should be used to interact directly with the LLM API, i.e.,
        for everything that is not related to the generation of an utterance.

        Args:
            prompt: Prompt for the LLM.
            initial_prompt: Initial prompt for the chat API. Defaults to None.

        Returns:
            Response from the LLM API without any post-processing.
        """
        if self.use_chat_api:
            messages = [
                {"role": "system", "content": initial_prompt},
                *self._parse_prompt_context(prompt),
            ]
            response = (
                self.client.chat.completions.create(
                    messages=messages, model=self.model, **self._llm_options  # type: ignore[arg-type] # noqa
                )
                .choices[0]
                .message.content
            )
        else:
            response = (
                self.client.completions.create(
                    model=self.model, prompt=prompt, **self._llm_options
                )
                .choices[0]
                .text
            )

        return response
