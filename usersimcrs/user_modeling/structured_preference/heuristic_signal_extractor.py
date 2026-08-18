"""Heuristic signal extractor for structured preference modeling."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.user_modeling.structured_preference.signal_extractor import (
    PreferenceSignal,
    PreferenceSignalsExtractor,
)


class HeuristicPreferenceSignalsExtractor(PreferenceSignalsExtractor):
    """Extracts preference signals with simple text heuristics."""

    PREFERENCE_SCORE = 0.7
    NEGATIVE_PATTERNS = (
        r"\b(?:avoid|nothing|not|no|without|not into|not too|too much)\b.*"
        r"\b{value}\b",
        r"\bnot interested in\b.*\b{value}\b",
        r"\b(?:anything but|other than|except|rather than)\b.*\b{value}\b",
    )
    POSITIVE_PATTERNS = (
        r"\b(?:like|love|prefer|enjoy)\b.*\b{value}\b",
        r"\blooking for\b.*\b{value}\b",
        r"\binterested in\b.*\b{value}\b",
        r"\b(?:with|about|featuring|set in)\b.*\b{value}\b",
        r"\bmore\b.*\b{value}\b",
    )

    def __init__(
        self,
        domain: SimulationDomain,
        item_collection: ItemCollection,
    ) -> None:
        """Initializes the heuristic signal extractor.

        Args:
            domain: Simulation domain.
            item_collection: Item collection.
        """
        self._domain = domain
        self._item_collection = item_collection
        self._catalog_slot_values = self._build_catalog_slot_values()
        self._item_names = self._build_item_names()

    def _build_catalog_slot_values(self) -> Dict[str, List[Tuple[str, str]]]:
        """Builds catalog slot values and normalized forms.

        Returns:
            Mapping from slot to original and normalized values.
        """
        catalog_slot_values: Dict[str, List[Tuple[str, str]]] = {}
        for slot in self._domain.get_slot_names():
            if slot.upper() in {"TITLE", "NAME"}:
                continue
            values = {}
            for value in self._item_collection.get_possible_property_values(
                slot
            ):
                if value is None:
                    continue
                normalized_value = self.normalize_preference_value(value)
                if len(normalized_value) >= 3:
                    values[normalized_value] = str(value)
            catalog_slot_values[slot] = [
                (value, normalized_value)
                for normalized_value, value in sorted(
                    values.items(), key=lambda item: len(item[0]), reverse=True
                )
            ]
        return catalog_slot_values

    def _build_item_names(self) -> Dict[str, str]:
        """Builds normalized item titles and names.

        Returns:
            Mapping from item identifier to normalized item name.
        """
        item_names: Dict[str, str] = {}
        available_slots = [
            slot
            for slot in ("TITLE", "NAME")
            if slot in self._domain.get_slot_names()
        ]
        if not available_slots:
            return item_names

        query = (
            f"SELECT id, {', '.join(available_slots)} "
            f"FROM {self._item_collection._table_name}"
        )
        self._item_collection._cursor.execute(query)
        for row in self._item_collection._cursor.fetchall():
            for slot in available_slots:
                value = row[slot]
                if value is None:
                    continue
                normalized = self.normalize_preference_value(value)
                if len(normalized) < 3:
                    continue
                item_names[row["id"]] = normalized
                break
        return item_names

    def _score_value_in_text(self, text: str, value: str) -> float:
        """Scores a matched value in text.

        Args:
            text: Normalized text.
            value: Normalized matched value.

        Returns:
            Preference score.
        """
        contexts = [
            clause.strip()
            for clause in re.split(r"[.!?;]", text)
            if value in clause
        ]
        for context in contexts:
            if any(
                re.search(pattern.format(value=re.escape(value)), context)
                for pattern in self.NEGATIVE_PATTERNS
            ):
                return -self.PREFERENCE_SCORE
            if any(
                re.search(pattern.format(value=re.escape(value)), context)
                for pattern in self.POSITIVE_PATTERNS
            ):
                return self.PREFERENCE_SCORE
        return 0

    def extract(self, user_utterance: str) -> List[PreferenceSignal]:
        """Extracts preference signals from user text.

        Args:
            user_utterance: User utterance.

        Returns:
            Extracted preference signals.
        """
        text = self.normalize_preference_value(user_utterance)
        if not text:
            return []
        signals: List[PreferenceSignal] = []

        for slot, values in self._catalog_slot_values.items():
            for value, normalized_value in values:
                if normalized_value not in text:
                    continue
                score = self._score_value_in_text(text, normalized_value)
                if score:
                    signals.append(
                        PreferenceSignal(
                            slot=slot,
                            value=value,
                            score=score,
                            source="attribute",
                        )
                    )

        for item_id, item_name in self._item_names.items():
            if item_name not in text:
                continue
            score = self._score_value_in_text(text, item_name)
            if score:
                signals.append(
                    PreferenceSignal(
                        item_id=item_id,
                        score=score,
                        source="item",
                    )
                )

        return signals
