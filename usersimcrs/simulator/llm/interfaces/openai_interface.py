"""Interface to use a LLM served by OpenAI."""

import os
import re
from typing import Dict, List

import yaml
from openai import OpenAI

from dialoguekit.core import Utterance
from dialoguekit.participant.participant import DialogueParticipant
from usersimcrs.simulator.llm.interfaces.llm_interface import LLMInterface
from usersimcrs.simulator.llm.prompt import Prompt


class OpenAILLMInterface(LLMInterface):
    def __init__(
        self,
        configuration_path: str,
        chat_api: bool = False,
        default_response: str = None,
    ) -> None:
        """Initializes interface for OpenAI served LLM.

        Args:
            configuration_path: Path to the configuration file.
            chat_api: Whether to use the chat or completion API. Defaults to
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
        self.chat_api = chat_api

    def generate_response(self, prompt: Prompt) -> Utterance:
        """Generates a user utterance given a prompt.

        Args:
            prompt: Prompt for generating the utterance.

        Returns:
            Utterance.
        """
        if self.chat_api:
            return self._generate_chat_response(prompt)

        return self._generate_completion_response(prompt.prompt_text)

    def _generate_chat_response(self, prompt: Prompt) -> Utterance:
        """Generates a user utterance using the chat API.

        Args:
            prompt: Prompt for generating the utterance.

        Returns:
            Utterance.
        """
        messages = [
            {"role": "system", "content": prompt.build_new_prompt()},
            *self._parse_prompt_context(prompt.prompt_text),
        ]
        response = (
            self.client.chat.completions.create(
                messages=messages, model=self.model, **self._llm_options  # type: ignore[arg-type] # noqa
            )
            .choices[0]
            .message.content
        )
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

    def _generate_completion_response(self, prompt: str) -> Utterance:
        """Generates a user utterance using the completion API.

        Args:
            prompt: Prompt for generating the utterance.

        Returns:
            Utterance.
        """
        response = (
            self.client.completions.create(
                model=self.model, prompt=prompt, **self._llm_options
            )
            .choices[0]
            .text
        )
        return Utterance(response, DialogueParticipant.USER)
