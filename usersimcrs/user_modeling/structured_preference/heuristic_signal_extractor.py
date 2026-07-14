"""Heuristic signal extractor for structured preference modeling."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.user_modeling.structured_preference.signal_extractor import (
    PreferenceSignal,
    PreferenceSignalsExtractor,
    normalize_preference_value,
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

    def _build_catalog_slot_values(self) -> Dict[str, List[str]]:
        """Builds normalized catalog slot values.

        Returns:
            Mapping from slot to normalized values.
        """
        catalog_slot_values: Dict[str, List[str]] = {}
        for slot in self._domain.get_slot_names():
            if slot.upper() in {"TITLE", "NAME"}:
                continue
            values = {
                normalize_preference_value(value)
                for value in self._item_collection.get_possible_property_values(
                    slot
                )
                if value is not None
                and len(normalize_preference_value(value)) >= 3
            }
            catalog_slot_values[slot] = sorted(values, key=len, reverse=True)
        return catalog_slot_values

    def _build_item_names(self) -> Dict[str, List[Tuple[str, str]]]:
        """Builds normalized item titles and names.

        Returns:
            Mapping from normalized item name to item identifiers.
        """
        item_names: Dict[str, List[Tuple[str, str]]] = {}
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
                normalized = normalize_preference_value(value)
                if len(normalized) < 3:
                    continue
                item_names.setdefault(row["id"], []).append((slot, normalized))
        return item_names

    def _extract_matched_values(self, text: str) -> List[Tuple[str, str]]:
        """Extracts matched slot values from text.

        Args:
            text: Normalized text.

        Returns:
            Matched slot-value pairs.
        """
        matches: List[Tuple[str, str]] = []
        for slot, values in self._catalog_slot_values.items():
            for value in values:
                if value in text:
                    matches.append((slot, value))
        return self._deduplicate_matches(matches)

    def _extract_matched_items(self, text: str) -> List[str]:
        """Extracts matched item identifiers from text.

        Args:
            text: Normalized text.

        Returns:
            Matched item identifiers.
        """
        matches: List[Tuple[str, str]] = []
        for item_id, item_name in self._item_names.items():
            if item_name and item_name[0][1] in text:
                matches.append((item_id, item_name[0][1]))
        return [item_id for item_id, _ in self._deduplicate_matches(matches)]

    def _deduplicate_matches(
        self, matches: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """Removes shorter duplicate matches.

        Args:
            matches: Raw matches.

        Returns:
            Filtered matches.
        """
        filtered: List[Tuple[str, str]] = []
        for key, value in sorted(matches, key=lambda match: -len(match[1])):
            if any(value != kept and value in kept for _, kept in filtered):
                continue
            filtered.append((key, value))
        return filtered

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

    def extract(
        self,
        user_utterance: str,
        rating: float | None = None,
        past_dialogues: List[str] | None = None,
    ) -> List[PreferenceSignal]:
        text = normalize_preference_value(user_utterance)
        signals: List[PreferenceSignal] = []

        for slot, value in self._extract_matched_values(text):
            score = self._score_value_in_text(text, value)
            if score:
                signals.append(
                    PreferenceSignal(
                        slot=slot,
                        value=value,
                        score=score,
                        source="attribute",
                    )
                )

        for item_id in self._extract_matched_items(text):
            item = self._item_collection.get_item(item_id)
            if item is None:
                continue
            item_name = next(
                (
                    normalize_preference_value(item.get_property(slot))
                    for slot in ("TITLE", "NAME")
                    if item.get_property(slot)
                ),
                "",
            )
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
