"""Preference update agent for structured preference modeling."""

from __future__ import annotations

from typing import Iterable, Set, Tuple

from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.user_modeling.structured_preference.preference_memory import (
    PreferenceMemory,
)
from usersimcrs.user_modeling.structured_preference.signal_extractor import (
    PreferenceSignal,
    normalize_preference_value,
)


class PreferenceUpdateAgent:
    """Applies extracted preference signals to preference memory."""

    def __init__(
        self,
        domain: SimulationDomain,
        item_collection: ItemCollection,
        long_term_preferences: PreferenceMemory,
        session_preferences: PreferenceMemory,
        preference_threshold: float,
        update_step: float,
        promotion_min_confirmations: int,
    ) -> None:
        """Initializes the preference update agent.

        Args:
            domain: Simulation domain.
            item_collection: Item collection.
            long_term_preferences: Long-term preference memory.
            session_preferences: Session preference memory.
            preference_threshold: Preference threshold.
            update_step: Update step size.
            promotion_min_confirmations: Minimum confirmations for promotion.
        """
        self._domain = domain
        self._item_collection = item_collection
        self._long_term_preferences = long_term_preferences
        self._session_preferences = session_preferences
        self._preference_threshold = preference_threshold
        self._update_step = update_step
        self._promotion_min_confirmations = promotion_min_confirmations

    def update_slot_value_preference(
        self, slot: str, value: str, score: float
    ) -> None:
        """Updates a slot-value preference.

        Args:
            slot: Slot name.
            value: Slot value.
            score: Preference score.
        """
        value = normalize_preference_value(value)
        if self._long_term_preferences.get(slot, value) is not None:
            self._update_existing_preference(
                self._long_term_preferences, slot, value, score
            )
            return
        self._update_session_preference(slot, value, score)

    def _update_existing_preference(
        self,
        preference_memory: PreferenceMemory,
        slot: str,
        value: str,
        score: float,
    ) -> None:
        """Updates an existing preference.

        Args:
            preference_memory: Preference memory to update.
            slot: Slot name.
            value: Slot value.
            score: Preference score.
        """
        preference_memory.increment_towards(
            slot, value, score, self._update_step
        )

    def _update_session_preference(
        self, slot: str, value: str, score: float
    ) -> None:
        """Updates a session preference.

        Args:
            slot: Slot name.
            value: Slot value.
            score: Preference score.
        """
        current = self._session_preferences.get(slot, value)
        if current is None:
            initial_score = (
                score
                if abs(score) < self._preference_threshold
                else (
                    self._update_step * 2
                    if score > 0
                    else -self._update_step * 2
                )
            )
            self._session_preferences.set(slot, value, initial_score, 1)
            return
        self._update_existing_preference(
            self._session_preferences, slot, value, score
        )

    def _item_signals_to_slot_signals(
        self, signals: Iterable[PreferenceSignal]
    ) -> list[PreferenceSignal]:
        """Converts item signals into slot-value signals.

        Args:
            signals: Item signals.

        Returns:
            Slot-value signals inferred from items.
        """
        slot_signals: list[PreferenceSignal] = []
        for signal in signals:
            if not signal.item_id:
                continue
            item = self._item_collection.get_item(signal.item_id)
            if item is None:
                continue
            for slot in self._domain.get_informable_slots():
                value = item.get_property(slot)
                if value is None:
                    continue
                values = value if isinstance(value, list) else [value]
                for entry in values:
                    slot_signals.append(
                        PreferenceSignal(
                            slot=slot,
                            value=normalize_preference_value(entry),
                            score=signal.score,
                            source="item",
                        )
                    )
        return slot_signals

    def _combine_signals(
        self, signals: Iterable[PreferenceSignal]
    ) -> list[PreferenceSignal]:
        """Combines repeated signals for the same slot-value pair.

        Args:
            signals: Signals to combine.

        Returns:
            Combined signals.
        """
        combined: dict[Tuple[str, str], PreferenceSignal] = {}
        for signal in signals:
            if not signal.slot or signal.value is None:
                continue
            key = (signal.slot, signal.value)
            if key not in combined:
                combined[key] = PreferenceSignal(
                    slot=signal.slot,
                    value=signal.value,
                    score=signal.score,
                    source=signal.source,
                )
                continue
            combined[key].score += signal.score
            if combined[key].source != "attribute":
                combined[key].source = signal.source
        return list(combined.values())

    def apply(self, signals: Iterable[PreferenceSignal]) -> None:
        """Applies extracted signals.

        Args:
            signals: Signals to apply.
        """
        explicit_signals = [
            signal
            for signal in signals
            if signal.slot and signal.value is not None
        ]
        explicit_keys: Set[Tuple[str, str]] = {
            (signal.slot, signal.value) for signal in explicit_signals
        }

        item_signals = [
            signal
            for signal in self._item_signals_to_slot_signals(signals)
            if (signal.slot, signal.value) not in explicit_keys
        ]

        for signal in self._combine_signals([*explicit_signals, *item_signals]):
            self.update_slot_value_preference(
                signal.slot,
                signal.value,
                max(-1.0, min(1.0, signal.score)),
            )

    def promote_session_preferences_to_long_term(self) -> None:
        """Promotes repeated session preferences to long-term memory."""
        for slot, value, score, count in self._session_preferences.items():
            if count < self._promotion_min_confirmations:
                continue
            if self._long_term_preferences.get(slot, value) is None:
                self._long_term_preferences.set(slot, value, score, count)
                continue
            self._update_existing_preference(
                self._long_term_preferences, slot, value, score
            )
            updated_count = (
                self._long_term_preferences.get_count(slot, value) + count
            )
            self._long_term_preferences.set(
                slot,
                value,
                self._long_term_preferences.get(slot, value),
                updated_count,
            )
