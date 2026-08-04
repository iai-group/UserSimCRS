"""Heuristic information need tracker.

The tracker inspects utterance text and dialogue-act annotations for mentions of
the slots and values contained in an information need. Slot mentions mark the
slot as attempted, while matching target or constraint values mark it complete.
"""

from __future__ import annotations

import re
from typing import Any, List

from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.utterance import Utterance

from usersimcrs.information_need_management.information_need import (
    InformationNeed,
)
from usersimcrs.information_need_management.information_need_tracker import (
    InformationNeedTracker,
    SlotUpdate,
)


class HeuristicInformationNeedTracker(InformationNeedTracker):
    def _normalize_text(self, text: str) -> str:
        """Normalizes text for matching.

        Args:
            text: Text to normalize.

        Returns:
            Normalized text.
        """
        return " ".join(re.sub(r"[_-]", " ", text.lower()).split())

    def _get_target_slot_values(self, slot: str) -> List[str]:
        """Collects normalized target values for a slot.

        Args:
            slot: Slot name.

        Returns:
            Normalized target values.
        """
        normalized_values: List[str] = []
        information_need = self.get_information_need()
        for item in information_need.target_items:
            value = item.get_property(slot)
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            normalized_values.extend(
                self._normalize_text(str(entry)) for entry in values
            )
        return normalized_values

    def _slot_update(
        self, kind: str, slot: str, is_complete: bool, value: Any = None
    ) -> SlotUpdate:
        """Builds a slot update with complete or attempted status.

        Args:
            kind: Slot kind, either request or constraint.
            slot: Slot name.
            is_complete: Whether the slot is complete.
            value: Slot value for request completion.

        Returns:
            Slot update.
        """
        return SlotUpdate(
            kind,
            slot,
            (
                InformationNeed.SLOT_STATE_COMPLETE
                if is_complete
                else InformationNeed.SLOT_STATE_ATTEMPTED
            ),
            value,
        )

    def update_from_utterance(self, utterance: Utterance) -> List[SlotUpdate]:
        """Builds information-need updates from an utterance.

        Args:
            utterance: Utterance to inspect.

        Returns:
            List of slot updates.
        """
        raw = getattr(utterance, "text", "")
        text = self._normalize_text(raw)

        if not text:
            return []

        updates: List[SlotUpdate] = []

        information_need = self.get_information_need()
        for slot in information_need.request_states:
            normalized_slot = self._normalize_text(slot)
            target_values = self._get_target_slot_values(slot)
            value_mentioned = any(value in text for value in target_values)
            slot_mentioned = normalized_slot in text
            if slot_mentioned or value_mentioned:
                updates.append(
                    self._slot_update(
                        "request",
                        slot,
                        value_mentioned,
                        raw.strip() if value_mentioned else None,
                    )
                )

        for slot, value in information_need.constraints.items():
            normalized_slot = self._normalize_text(slot)
            values = value if isinstance(value, list) else [value]
            normalized_values = [self._normalize_text(str(v)) for v in values]
            slot_mentioned = normalized_slot in text
            value_mentioned = any(v in text for v in normalized_values)

            if slot_mentioned or value_mentioned:
                updates.append(
                    self._slot_update(
                        "constraint",
                        slot,
                        value_mentioned,
                    )
                )

        return updates

    def update_from_agent_dialogue_acts(
        self,
        agent_dialogue_acts: List[DialogueAct],
    ) -> List[SlotUpdate]:
        """Builds information-need updates from agent dialogue acts.

        Args:
            agent_dialogue_acts: Agent dialogue acts to inspect.

        Returns:
            List of slot updates.
        """
        updates: List[SlotUpdate] = []
        for dialogue_act in agent_dialogue_acts:
            for annotation in dialogue_act.annotations:
                slot = annotation.slot
                value = annotation.value
                information_need = self.get_information_need()
                if slot in information_need.request_states:
                    updates.append(
                        self._slot_update(
                            "request", slot, value is not None, value
                        )
                    )

                if slot in information_need.constraint_states:
                    updates.append(
                        self._slot_update(
                            "constraint",
                            slot,
                            value
                            == information_need.get_constraint_value(slot),
                        ),
                    )
        return updates
