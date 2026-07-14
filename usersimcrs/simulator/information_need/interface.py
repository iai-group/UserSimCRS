"""Information-need update interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List

from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.utterance import Utterance

from usersimcrs.core.information_need import InformationNeed


@dataclass
class SlotUpdate:
    kind: str
    slot: str
    status: str
    value: Any = None


class InformationNeedInterface(ABC):
    @abstractmethod
    def update_from_utterance(
        self, information_need: InformationNeed, utterance: Utterance
    ) -> List[SlotUpdate]:
        """Builds information-need updates from an utterance.

        Args:
            information_need: Information need to update.
            utterance: Utterance to inspect.

        Return:
            List of slot updates.
        """
        raise NotImplementedError

    @abstractmethod
    def update_from_agent_dialogue_acts(
        self,
        information_need: InformationNeed,
        agent_dialogue_acts: List[DialogueAct],
    ) -> List[SlotUpdate]:
        """Builds information-need updates from agent dialogue acts.

        Args:
            information_need: Information need to update.
            agent_dialogue_acts: Agent dialogue acts to inspect.

        Return:
            List of slot updates.
        """
        raise NotImplementedError


def apply_information_need_update(
    information_need: InformationNeed, updates: List[SlotUpdate]
) -> None:
    """Applies slot updates to an information need.

    Args:
        information_need: Information need to update.
        updates: Slot updates to apply.
    """
    for update in updates:
        if update.kind == "request":
            if update.status == InformationNeed.SLOT_STATE_COMPLETE:
                information_need.mark_request_complete(
                    update.slot, update.value
                )
            elif update.status == InformationNeed.SLOT_STATE_ATTEMPTED:
                information_need.mark_request_attempted(update.slot)
        elif update.status == InformationNeed.SLOT_STATE_COMPLETE:
            information_need.mark_constraint_complete(update.slot)
        elif update.status == InformationNeed.SLOT_STATE_ATTEMPTED:
            information_need.mark_constraint_attempted(update.slot)
