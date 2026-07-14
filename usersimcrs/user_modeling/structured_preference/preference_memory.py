"""Preference memory helpers for structured preference modeling."""

from __future__ import annotations

from typing import Dict, Iterator, Tuple

from dialoguekit.participant.user_preferences import UserPreferences


class PreferenceMemory:
    """Stores slot-value preferences and their evidence counts."""

    def __init__(self, user_id: str) -> None:
        """Initializes preference memory.

        Args:
            user_id: User identifier.
        """
        self._user_id = user_id
        self._preferences = UserPreferences(user_id)
        self._counts: Dict[Tuple[str, str], int] = {}

    def get(self, slot: str, value: str) -> float | None:
        """Returns a stored preference.

        Args:
            slot: Slot name.
            value: Slot value.

        Returns:
            Stored preference or None.
        """
        return self._preferences.get_preference(slot, value)

    def set(
        self, slot: str, value: str, score: float, count: int | None = None
    ) -> None:
        """Stores a preference.

        Args:
            slot: Slot name.
            value: Slot value.
            score: Preference score.
            count: Optional evidence count.
        """
        self._preferences.set_preference(slot, value, score)
        if count is not None:
            self._counts[(slot, value)] = count

    def increment_towards(
        self, slot: str, value: str, score: float, step: float
    ) -> float:
        """Moves a stored preference toward a new signal.

        Args:
            slot: Slot name.
            value: Slot value.
            score: New signal score.
            step: Update step size.

        Returns:
            Updated preference score.
        """
        current = self.get(slot, value)
        if current is None:
            self.set(slot, value, score, 1)
            return score

        direction = 1 if score > current else -1
        updated = max(-1.0, min(1.0, current + direction * step))
        self.set(slot, value, updated)
        self._counts[(slot, value)] = self.get_count(slot, value) + 1
        return updated

    def get_count(self, slot: str, value: str) -> int:
        """Returns evidence count for a preference.

        Args:
            slot: Slot name.
            value: Slot value.

        Returns:
            Evidence count.
        """
        return self._counts.get((slot, value), 0)

    def clear(self) -> None:
        """Clears all stored preferences."""
        self._preferences = UserPreferences(self._user_id)
        self._counts.clear()

    def items(self) -> Iterator[Tuple[str, str, float, int]]:
        """Iterates over stored preferences.

        Returns:
            Iterator of slot, value, score, and evidence count.
        """
        for slot, value_preferences in self._preferences._preferences.items():
            for value, score in value_preferences.items():
                yield slot, value, score, self.get_count(slot, value)
