"""Annotate information need based on annotated dialogue.

This annotator uses the annotated dialogue to identify a possible information
need that the user had during the conversation. The information need comprises
constraints and requests that user expressed during the conversation.
"""

import json
import logging
import os

import yaml
from ollama import Client, Options

from dialoguekit.core.dialogue import Dialogue
from usersimcrs.core.information_need import InformationNeed

DEFAULT_INITIAL_PROMPT_FILE = (
    "scripts/redial/information_need_prompt_default.txt"
)


class InformationNeedAnnotator:
    def __init__(
        self,
        configuration_file: str,
        prompt_file: str = DEFAULT_INITIAL_PROMPT_FILE,
    ) -> None:
        """Initializes the annotator.

        Args:
            configuration_file: Configuration file for Ollama.
            prompt_file: File containing prompt, it should have a placeholder
              for the dialogue. Defaults to DEFAULT_INITIAL_PROMPT_FILE.

        Raises:
            FileNotFoundError: If the configuration file is not found.
        """
        if not os.path.exists(configuration_file):
            raise FileNotFoundError(
                f"No configuration file: {configuration_file}"
            )

        configuration = yaml.safe_load(open(configuration_file))

        # Ollama
        self.client = Client(host=configuration.get("host"))
        self._model = configuration.get("model")
        self._options = Options(**configuration.get("options", {}))

        # Prompt
        self.prompt = open(prompt_file).read()

    def annotate_information_need(self, dialogue: Dialogue) -> Dialogue:
        """Annotates information need in the dialogue.

        Args:
            dialogue: Dialogue to annotate.

        Returns:
            Dialogue with annotated information need.
        """
        json_dialogue = json.dumps(dialogue.to_dict(), indent=2)
        prompt = self.prompt.replace("{dialogue}", json_dialogue)

        response = self.client.generate(
            prompt=prompt, model=self._model, options=self._options
        ).get("response", "")

        try:
            information_need = self.parse_model_output(response)
            dialogue.metadata["information_need"] = information_need.to_dict()
        except Exception as e:
            logging.error(
                "Failed to parse model output for dialogue "
                f"{dialogue.conversation_id}: {response}\n{e}"
            )
        return dialogue

    def parse_model_output(self, response: str) -> InformationNeed:
        """Parses the model output to extract information need.

        Args:
            response: Model output.

        Returns:
            Extracted information need.
        """
        json_response = json.loads(response)
        json_response["target_items"] = []
        return InformationNeed.from_dict(json_response)
