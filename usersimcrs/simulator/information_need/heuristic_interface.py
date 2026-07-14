"""Heuristic information-need update interface."""

from __future__ import annotations

import re
from typing import List

from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.utterance import Utterance

from usersimcrs.core.information_need import InformationNeed
from usersimcrs.simulator.information_need.interface import (
    InformationNeedInterface,
    SlotUpdate,
)


class HeuristicInformationNeedInterface(InformationNeedInterface):
    def _normalize_text(self, text: str) -> str:
        return " ".join(re.sub(r"[_-]", " ", text.lower()).split())

    def _get_target_slot_values(
        self, information_need: InformationNeed, slot: str
    ) -> List[str]:
        normalized_values: List[str] = []
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
        self, kind: str, slot: str, is_complete: bool, value=None
    ) -> SlotUpdate:
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

    def update_from_utterance(
        self, information_need: InformationNeed, utterance: Utterance
    ) -> List[SlotUpdate]:
        raw = getattr(utterance, "text", "")
        text = self._normalize_text(raw)

        if not text:
            return []

        updates: List[SlotUpdate] = []

        for slot in information_need.request_states:
            normalized_slot = self._normalize_text(slot)
            target_values = self._get_target_slot_values(information_need, slot)
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
        information_need: InformationNeed,
        agent_dialogue_acts: List[DialogueAct],
    ) -> List[SlotUpdate]:
        updates: List[SlotUpdate] = []
        for dialogue_act in agent_dialogue_acts:
            for annotation in dialogue_act.annotations:
                slot = annotation.slot
                value = annotation.value
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
