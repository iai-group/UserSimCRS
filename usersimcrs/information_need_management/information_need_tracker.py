"""Information need tracker.

It updates the state of the information need during the conversation. That is,
tracking the state (incomplete, attempted, complete) of the constraints and
requests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List

from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.utterance import Utterance

from usersimcrs.information_need_management.information_need import (
    InformationNeed,
)


@dataclass
class SlotUpdate:
    kind: str
    slot: str
    status: str
    value: Any = None


class InformationNeedTracker(ABC):
    def __init__(self, information_need: InformationNeed) -> None:
        """Initializes the tracker with the information need to update."""
        self._information_need = information_need

    def set_information_need(self, information_need: InformationNeed) -> None:
        """Sets the information need tracked by this tracker.

        Args:
            information_need: Information need to track.
        """
        self._information_need = information_need

    def get_information_need(self) -> InformationNeed:
        """Returns the tracked information need.

        Returns:
            Information need.
        """
        return self._information_need

    @abstractmethod
    def update_from_utterance(self, utterance: Utterance) -> List[SlotUpdate]:
        """Builds information-need updates from an utterance.

        Args:
            utterance: Utterance to inspect.

        Returns:
            List of slot updates.

        Raises:
            NotImplementedError: If not implemented in derived class.
        """
        raise NotImplementedError

    @abstractmethod
    def update_from_agent_dialogue_acts(
        self, agent_dialogue_acts: List[DialogueAct]
    ) -> List[SlotUpdate]:
        """Builds information-need updates from agent dialogue acts.

        Args:
            agent_dialogue_acts: Agent dialogue acts to inspect.

        Returns:
            List of slot updates.

        Raises:
            NotImplementedError: If not implemented in derived class.
        """
        raise NotImplementedError

    def apply_updates_to_information_need(
        self, updates: List[SlotUpdate]
    ) -> None:
        """Applies slot updates to the tracked information need.

        Args:
            updates: Slot updates to apply to the tracked information need.
        """
        for update in updates:
            if update.kind == "request":
                if update.status == InformationNeed.SLOT_STATE_COMPLETE:
                    self._information_need.mark_request_complete(
                        update.slot, update.value
                    )
                elif update.status == InformationNeed.SLOT_STATE_ATTEMPTED:
                    self._information_need.mark_request_attempted(update.slot)
            elif update.status == InformationNeed.SLOT_STATE_COMPLETE:
                self._information_need.mark_constraint_complete(update.slot)
            elif update.status == InformationNeed.SLOT_STATE_ATTEMPTED:
                self._information_need.mark_constraint_attempted(update.slot)
