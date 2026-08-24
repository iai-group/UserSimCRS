"""Preference memory helpers for structured preference modeling."""

from __future__ import annotations

from typing import Dict, List, Tuple

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
        self._counts: Dict[Tuple[str, str], float] = {}

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
        self, slot: str, value: str, score: float, count: float | None = None
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
    ) -> None:
        """Moves a stored preference toward a new signal.

        Args:
            slot: Slot name.
            value: Slot value.
            score: New signal score.
            step: Update step size.
        """
        current = self.get(slot, value)
        if current is None:
            self.set(slot, value, score, 1)
            return

        count = self.get_count(slot, value)
        weighted_step = step / max(1, count)
        updated = current + weighted_step * (score - current)
        updated = max(-1.0, min(1.0, updated))
        self.set(slot, value, updated, count + 1)

    def get_count(self, slot: str, value: str) -> float:
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

    def items(self) -> List[Tuple[str, str, float, float]]:
        """Returns stored preferences.

        Returns:
            List of slot, value, score, and evidence count.
        """
        preferences = self._preferences._preferences.items()
        return [
            (slot, value, score, self.get_count(slot, value))
            for slot, value_preferences in preferences
            for value, score in value_preferences.items()
        ]

    def ranked_preferences(self) -> List[Tuple[str, str, float]]:
        """Returns preferences ranked by evidence count and score."""
        return sorted(
            [(slot, value, score) for slot, value, score, _ in self.items()],
            key=lambda preference: (
                -self.get_count(preference[0], preference[1]),
                -abs(preference[2]),
                preference[0],
                preference[1],
            ),
        )

    def set_weighted_scores(
        self,
        weighted_sources: List[
            Tuple[Dict[Tuple[str, str], List[float]], float]
        ],
        preference_threshold: float,
    ) -> None:
        """Stores weighted scores from historical sources.

        Args:
            weighted_sources: Score dictionaries paired with source weights.
            preference_threshold: Minimum absolute score to store.
        """
        keys: set[Tuple[str, str]] = set()
        for scores, _ in weighted_sources:
            keys.update(scores)

        for slot, value in keys:
            weighted_count = sum(
                weight * len(scores.get((slot, value), []))
                for scores, weight in weighted_sources
            )
            if weighted_count <= 0:
                continue

            score = (
                sum(
                    weight * sum(scores.get((slot, value), []))
                    for scores, weight in weighted_sources
                )
                / weighted_count
            )
            if abs(score) >= preference_threshold:
                self.set(slot, value, score, weighted_count)
