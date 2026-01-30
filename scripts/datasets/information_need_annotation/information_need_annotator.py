"""Annotate information need based on annotated dialogue.

This annotator uses the annotated dialogue to identify a possible information
need that the user had during the conversation. The information need comprises
constraints and requests that user expressed during the conversation.
"""

import json
import logging
import os

from dialoguekit.core.dialogue import Dialogue
from usersimcrs.core.information_need import InformationNeed
from usersimcrs.llm_interfaces.llm_interface import LLMInterface

DEFAULT_INITIAL_PROMPT_MOVIES_FILE = "scripts/datasets/information_need_annotation/information_need_prompt_movies_default.txt"  # noqa: E501


class InformationNeedAnnotator:
    def __init__(
        self,
        llm_interface: LLMInterface,
        prompt_file: str = DEFAULT_INITIAL_PROMPT_MOVIES_FILE,
    ) -> None:
        """Initializes the annotator.

        Args:
            llm_interface: Interface to the large language model.
            prompt_file: File containing prompt, it should have a placeholder
              for the dialogue. Defaults to DEFAULT_INITIAL_PROMPT_MOVIES_FILE.

        Raises:
            FileNotFoundError: If the prompt file is not found.
        """
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(
                f"No file including the prompt: {prompt_file}"
            )

        self.llm_interface = llm_interface

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

        response = self.llm_interface.get_llm_api_response(prompt)

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
