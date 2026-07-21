"""LLM-backed information need tracker."""

from __future__ import annotations

from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.utterance import Utterance

from usersimcrs.information_need_management.information_need_tracker import (
    InformationNeedTracker,
    SlotUpdate,
)


class LLMInformationNeedTracker(InformationNeedTracker):
    def update_from_utterance(self, utterance: Utterance) -> list[SlotUpdate]:
        raise NotImplementedError

    def update_from_agent_dialogue_acts(
        self,
        agent_dialogue_acts: list[DialogueAct],
    ) -> list[SlotUpdate]:
        raise NotImplementedError
