"""Structured preference model backed by historical ratings and metadata."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.participant import DialogueParticipant
from dialoguekit.participant.user_preferences import UserPreferences

from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item import Item
from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.items.ratings import Ratings
from usersimcrs.user_modeling.preference_model import (
    KEY_ITEM_ID,
    PreferenceModel,
)
from usersimcrs.user_modeling.structured_preference import (
    HeuristicPreferenceSignalsExtractor,
    PreferenceMemory,
    PreferenceSignal,
    PreferenceSignalsExtractor,
)


class StructuredPreferenceModel(PreferenceModel):

    UPDATE_STEP = 0.25
    RATING_HISTORY_WEIGHT = 1.0
    DIALOGUE_HISTORY_WEIGHT = 1.0

    def __init__(
        self,
        domain: SimulationDomain,
        item_collection: ItemCollection,
        historical_ratings: Optional[Ratings] = None,
        historical_user_id: Optional[str] = None,
        preference_threshold: float = PreferenceModel.PREFERENCE_THRESHOLD,
        dialogue_history: Optional[List[Dialogue]] = None,
        signals_extractor: Optional[PreferenceSignalsExtractor] = None,
        rating_history_weight: float = RATING_HISTORY_WEIGHT,
        dialogue_history_weight: float = DIALOGUE_HISTORY_WEIGHT,
    ) -> None:
        """Initializes structured preferences.

        Args:
            domain: Domain.
            item_collection: Item collection.
            historical_ratings: Optional historical ratings. Defaults to None.
            historical_user_id: Historical user ID. Defaults to None.
            preference_threshold: Minimum absolute score to store as a
              meaningful preference. Defaults to
              `PreferenceModel.PREFERENCE_THRESHOLD`.
            dialogue_history: Optional previous dialogues to initialize session
              preferences from. Defaults to None.
            signals_extractor: Optional preference signal extractor. Defaults
              to None.
            rating_history_weight: Weight for evidence extracted from
              historical ratings. Defaults to 1.0.
            dialogue_history_weight: Weight for evidence extracted from
              historical dialogues. Defaults to 1.0.
        """
        historical_ratings = historical_ratings or Ratings(item_collection)
        super().__init__(
            domain, item_collection, historical_ratings, historical_user_id
        )
        self._preference_threshold = preference_threshold
        self._dialogue_history = list(dialogue_history or [])
        self._rating_history_weight = rating_history_weight
        self._dialogue_history_weight = dialogue_history_weight
        self._item_preferences = UserPreferences(self._user_id)
        self._long_term_preferences = PreferenceMemory(self._user_id)
        self._session_preferences = PreferenceMemory(self._user_id)
        self._signals_extractor = (
            signals_extractor
            or HeuristicPreferenceSignalsExtractor(domain, item_collection)
        )
        self._initialize_preferences()

    def _initialize_preferences(self) -> None:
        """Initializes preferences from historical ratings and dialogues."""
        self._long_term_preferences.set_weighted_scores(
            [
                (self._rating_scores(), self._rating_history_weight),
                (self._dialogue_scores(), self._dialogue_history_weight),
            ],
            self._preference_threshold,
        )

    def _rating_scores(self) -> Dict[Tuple[str, str], List[float]]:
        """Returns preference scores extracted from historical ratings.

        Returns:
            Rating scores keyed by slot-value pair.
        """
        rating_scores: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for item_id, rating in self._historical_ratings.get_user_ratings(
            self._historical_user_id
        ).items():
            item = self._item_collection.get_item(item_id)
            if item is None:
                continue
            if abs(rating) >= self._preference_threshold:
                self._item_preferences.set_preference(
                    KEY_ITEM_ID, item_id, rating
                )
            for slot, value in self._item_slot_values(item):
                rating_scores[(slot, value)].append(rating)
        return rating_scores

    def _dialogue_scores(self) -> Dict[Tuple[str, str], List[float]]:
        """Returns preference scores extracted from historical dialogues.

        Returns:
            Dialogue scores keyed by slot-value pair.
        """
        dialogue_scores: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for dialogue in self._dialogue_history:
            for utterance in self._user_utterances(dialogue):
                for signal in self._extract_slot_signals_from_text(
                    utterance.text
                ):
                    if not signal.slot or signal.value is None:
                        continue
                    dialogue_scores[(signal.slot, signal.value)].append(
                        signal.score
                    )
        return dialogue_scores

    def _user_utterances(self, dialogue: Dialogue) -> List[Utterance]:
        """Returns user utterances from a dialogue.

        Args:
            dialogue: Dialogue to read.

        Returns:
            User utterances.
        """
        return [
            utterance
            for utterance in dialogue.utterances
            if utterance.participant is DialogueParticipant.USER
        ]

    def _extract_slot_signals_from_text(
        self, text: str
    ) -> List[PreferenceSignal]:
        """Extracts slot-value preference signals from one text.

        Args:
            text: User utterance text.

        Returns:
            Slot-value preference signals.
        """
        text = text.strip()
        if not text:
            return []

        signals = self._signals_extractor.extract(text)
        return list(self._slot_preference_signals(signals))

    def _item_slot_values(self, item: Item) -> List[Tuple[str, str]]:
        """Returns preference slot values for an item.

        Args:
            item: Item to read.

        Returns:
            Slot-value pairs from informable item properties.
        """
        slot_values = []
        for slot in self._domain.get_informable_slots():
            value = item.get_property(slot)
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            for entry in values:
                slot_values.append((slot, str(entry)))
        return slot_values

    def _slot_preference_signals(
        self, signals: Iterable[PreferenceSignal]
    ) -> List[PreferenceSignal]:
        """Converts extracted signals to slot-value preference signals.

        Args:
            signals: Extracted preference signals.

        Returns:
            Slot-value preference signals.
        """
        explicit_keys = set()
        slot_signals = []
        for signal in signals:
            score = max(-1.0, min(1.0, signal.score))
            if signal.slot and signal.value is not None:
                explicit_keys.add((signal.slot, signal.value))
                slot_signals.append(
                    PreferenceSignal(
                        slot=signal.slot,
                        value=signal.value,
                        score=score,
                        source=signal.source,
                    )
                )
            if not signal.item_id:
                continue
            item = self._item_collection.get_item(signal.item_id)
            if item is None:
                continue
            for slot, value in self._item_slot_values(item):
                if (slot, value) in explicit_keys:
                    continue
                slot_signals.append(
                    PreferenceSignal(
                        slot=slot,
                        value=value,
                        score=score,
                        source=signal.source,
                    )
                )
        return slot_signals

    def get_item_preference(self, item_id: str) -> float:
        """Returns preference score for an item.

        Args:
            item_id: Item identifier.

        Returns:
            Long-term item preference if available, otherwise 0.
        """
        self._assert_item_exists(item_id)
        preference = self._item_preferences.get_preference(KEY_ITEM_ID, item_id)
        return preference if preference is not None else 0

    def get_slot_value_preference(self, slot: str, value: str) -> float:
        """Returns preference score for a slot-value pair.

        Session preferences are checked first because they capture the current
        dialogue context; long-term preferences are used as a fallback.

        Args:
            slot: Slot name.
            value: Slot value.

        Returns:
            Preference score for the slot-value pair, or 0 if unavailable.
        """
        self._assert_slot_exists(slot)
        preference = self._session_preferences.get(slot, value)
        if preference is None:
            preference = self._long_term_preferences.get(slot, value)
        return preference if preference is not None else 0

    def update_slot_value_preference(
        self, slot: str, value: str, score: float
    ) -> None:
        """Updates one slot-value preference.

        Args:
            slot: Slot name.
            value: Slot value.
            score: Preference score to store.
        """
        self._assert_slot_exists(slot)
        self._session_preferences.increment_towards(
            slot,
            value,
            max(-1.0, min(1.0, score)),
            self.UPDATE_STEP,
        )

    def get_preference_summary(self, max_preferences: int = 20) -> str:
        """Returns a preference summary for the LLM.

        Args:
            max_preferences: Total number of preferences. Defaults to 20.

        Returns:
            Summary for the LLM of session and long-term preferences.
        """
        session_preferences = self._session_preferences.ranked_preferences()[
            : max(1, max_preferences // 2)
        ]
        long_term_preferences = self._long_term_preferences.ranked_preferences()
        positive_preferences = [
            preference
            for preference in long_term_preferences
            if preference[2] >= self._preference_threshold
        ][: max(1, max_preferences // 2)]
        negative_preferences = [
            preference
            for preference in long_term_preferences
            if preference[2] <= -self._preference_threshold
        ][: max(1, max_preferences // 2)]

        sections = [
            ("Session preferences", session_preferences),
            ("Positive preferences", positive_preferences),
            ("Negative preferences", negative_preferences),
        ]
        return " | ".join(
            f"{title}: "
            + "; ".join(
                f"{slot}={value}:{score:.2f}"
                for slot, value, score in preferences
            )
            for title, preferences in sections
            if preferences
        )

    def update_from_utterance(self, utterance: Utterance) -> None:
        """Updates preferences from a user utterance.

        Args:
            utterance: User utterance used to update preferences.
        """
        for signal in self._extract_slot_signals_from_text(utterance.text):
            if not signal.slot or signal.value is None:
                continue
            self.update_slot_value_preference(
                signal.slot, signal.value, signal.score
            )
