"""LLM-backed information-need update interface."""

from __future__ import annotations

from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.utterance import Utterance

from usersimcrs.core.information_need import InformationNeed
from usersimcrs.simulator.information_need.interface import (
    InformationNeedInterface,
    SlotUpdate,
)


class LLMInformationNeedInterface(InformationNeedInterface):
    def update_from_utterance(
        self, information_need: InformationNeed, utterance: Utterance
    ) -> list[SlotUpdate]:
        raise NotImplementedError

    def update_from_agent_dialogue_acts(
        self,
        information_need: InformationNeed,
        agent_dialogue_acts: list[DialogueAct],
    ) -> list[SlotUpdate]:
        raise NotImplementedError
