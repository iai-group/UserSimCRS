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

    def ranked_items(self) -> List[Tuple[str, str, float]]:
        """Returns preferences ranked by evidence count and score.

        Returns:
            Ranked slot-value preferences.
        """
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
        primary_scores: Dict[Tuple[str, str], List[float]],
        secondary_scores: Dict[Tuple[str, str], List[float]],
        primary_weight: float,
        secondary_weight: float,
        preference_threshold: float,
    ) -> None:
        """Stores weighted scores from two historical sources.

        Args:
            primary_scores: Scores from the first source.
            secondary_scores: Scores from the second source.
            primary_weight: Weight for the first source.
            secondary_weight: Weight for the second source.
            preference_threshold: Minimum absolute score to store.
        """
        for slot, value in set(primary_scores).union(secondary_scores):
            primary = primary_scores.get((slot, value), [])
            secondary = secondary_scores.get((slot, value), [])
            primary_count = len(primary)
            secondary_count = len(secondary)
            weighted_count = (
                primary_weight * primary_count
                + secondary_weight * secondary_count
            )
            if weighted_count <= 0:
                continue

            score = (
                primary_weight * sum(primary)
                + secondary_weight * sum(secondary)
            ) / weighted_count
            if abs(score) >= preference_threshold:
                self.set(slot, value, score, weighted_count)
