"""LLM-backed information need tracker.

This module defines the prompt shape and class contract for an LLM-backed
tracker. The tracking logic is intentionally left unimplemented until the prompt
format and output parsing contract are finalized.
"""

from __future__ import annotations

from typing import List

from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.utterance import Utterance

from usersimcrs.information_need_management.information_need import (
    InformationNeed,
)
from usersimcrs.information_need_management.information_need_tracker import (
    InformationNeedTracker,
    SlotUpdate,
)
from usersimcrs.llm_interfaces.llm_interface import LLMInterface

DEFAULT_UPDATE_PROMPT = """\
You update the state of a simulated user's information need.

Information need:
{information_need}

Conversation evidence:
{evidence}

Return only JSON as a list of objects with keys:
- kind: "constraint" or "request"
- slot: the slot name
- status: "attempted" or "complete"
- value: the observed value, or null
"""


class LLMInformationNeedTracker(InformationNeedTracker):
    def __init__(
        self,
        information_need: InformationNeed,
        llm_interface: LLMInterface,
        prompt_template: str = DEFAULT_UPDATE_PROMPT,
    ) -> None:
        """Initializes the LLM-backed tracker.

        Args:
            information_need: Information need to track.
            llm_interface: Interface to the large language model.
            prompt_template: Prompt template containing information_need and
              evidence placeholders.
        """
        super().__init__(information_need)
        self.llm_interface = llm_interface
        self.prompt_template = prompt_template

    def update_from_utterance(self, utterance: Utterance) -> List[SlotUpdate]:
        """Builds information-need updates from an utterance.

        Args:
            utterance: Utterance to inspect.

        Returns:
            List of slot updates.

        Raises:
            NotImplementedError: Until the LLM tracking implementation is
              defined.
        """
        raise NotImplementedError

    def update_from_agent_dialogue_acts(
        self,
        agent_dialogue_acts: List[DialogueAct],
    ) -> List[SlotUpdate]:
        """Builds information-need updates from agent dialogue acts.

        Args:
            agent_dialogue_acts: Agent dialogue acts to inspect.

        Returns:
            List of slot updates.

        Raises:
            NotImplementedError: Until the LLM tracking implementation is
              defined.
        """
        raise NotImplementedError
