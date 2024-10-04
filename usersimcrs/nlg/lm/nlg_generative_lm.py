"""Generative natural language generation using a language model."""

import os
from typing import List, Optional, Union

import yaml
from ollama import Client, Options

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.annotation import Annotation
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.nlg.nlg_abstract import AbstractNLG
from dialoguekit.participant.participant import DialogueParticipant


class LMGenerativeNLG(AbstractNLG):
    def __init__(
        self,
        ollama_config_file: str,
        prompt_file: str,
        prompt_prefix: Optional[str] = None,
    ) -> None:
        """Initializes the generative NLG.

        Args:
            ollama_config_file: Path to the Ollama configuration file.
            prompt_file: Path to the prompt file.
            prompt_prefix: Prefix to be remove from generated utterances.
              Defaults to None.

        Raises:
            FileNotFoundError: If the Ollama configuration file or the prompt
              file is not found.
        """
        if not os.path.exists(ollama_config_file):
            raise FileNotFoundError(f"File '{ollama_config_file}' not found.")
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"File '{prompt_file}' not found.")

        # Ollama
        ollama_config = yaml.safe_load(open(ollama_config_file, "r"))
        self.client = Client(ollama_config.get("host"))
        self._model = ollama_config.get("model")
        self._stream = ollama_config.get("stream", False)
        self._options = Options(**ollama_config.get("options", {}))

        # Prompt
        self.prompt = open(prompt_file, "r").read()
        self.prompt_prefix = prompt_prefix

    def generate_utterance_text(
        self,
        dialogue_acts: List[DialogueAct],
        annotations: Optional[Union[List[Annotation], None]] = None,
        force_annotation: bool = False,
    ) -> Union[AnnotatedUtterance, bool]:
        """Turns a structured utterance into a textual one.

        Args:
            dialogue_acts: Dialogue acts of the utterance to be generated.
            annotations: If provided, these annotations should be considered
              during generation.
            force_annotation: A flag to indicate whether annotations should be
              forced or not. Not used in this NLG.

        Raises:
            RuntimeError: If generation fails.

        Returns:
            Natural language utterance.
        """
        try:
            dialogue_acts_str = self._stringify_dialogue_acts(dialogue_acts)

            if annotations:
                annonations_str = self._stringify_annotations(annotations)
                prompt = self.prompt.format(
                    dialogue_acts=dialogue_acts_str,
                    annotations=annonations_str,
                )
            else:
                prompt = self.prompt.format(dialogue_acts=dialogue_acts_str)

            response = self.client.generate(
                prompt=prompt,
                model=self._model,
                options=self._options,
                stream=self._stream,
            ).get("response", "")
            response = response.strip().replace(self.prompt_prefix, "")
        except Exception as e:
            raise RuntimeError(f"Failed to generate utterance: {e}")

        return AnnotatedUtterance(
            text=response,
            participant=DialogueParticipant.USER,
            dialogue_acts=dialogue_acts,
            annotations=annotations,
        )

    def _stringify_dialogue_acts(self, dialogue_acts: List[DialogueAct]) -> str:
        """Stringifies dialogue acts.

        The stringified dialogue acts are in the format:
        "intent(slot=value,...)|intent(slot,...)|..."

        Args:
            dialogue_acts: List of dialogue acts.

        Returns:
            List of dialogue acts as a string.
        """
        dialogue_acts_str = []
        for dialogue_act in dialogue_acts:
            dialogue_act_str = dialogue_act.intent.label
            if dialogue_act.annotations:
                annotations_str = ",".join(
                    [
                        f"{a.slot}={a.value}" if a.value else f"{a.slot}"
                        for a in dialogue_act.annotations
                    ]
                )
                dialogue_act_str = f"{dialogue_act_str}({annotations_str})"
            else:
                dialogue_act_str = f"{dialogue_act_str}()"
            dialogue_acts_str.append(dialogue_act_str)
        return "|".join(dialogue_acts_str)

    def _stringify_annotations(self, annotations: List[Annotation]) -> str:
        """Stringifies annotations.

        The stringified annotations are in the format:
        "slot=value\nslot\n..."

        Args:
            annotations: List of annotations.

        Returns:
            List of annotations as a string.
        """
        return "\n".join(
            [
                f"{a.slot}={a.value}" if a.value else f"{a.slot}"
                for a in annotations
            ]
        )
