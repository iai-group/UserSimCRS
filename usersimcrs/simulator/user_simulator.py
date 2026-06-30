"""User simulator abstract class."""

from abc import ABC, abstractmethod
import re

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.user import User, UserType
from usersimcrs.core.information_need import generate_random_information_need
from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item_collection import ItemCollection


class UserSimulator(User, ABC):
    def __init__(
        self,
        id: str,
        domain: SimulationDomain,
        item_collection: ItemCollection,
    ) -> None:
        """Initializes the user simulator."""
        super().__init__(id, UserType.SIMULATOR)
        self._domain = domain
        self._item_collection = item_collection
        self.get_new_information_need()

    def get_new_information_need(self) -> None:
        """Generates a new information need."""
        self.information_need = generate_random_information_need(
            self._domain, self._item_collection
        )

    def _normalize_text(self, text: str) -> str:
        """Normalizes text for simple slot and value matching.

        Lowercases the text, replaces underscores and hyphens with spaces,
        and collapses repeated whitespace.

        Args:
            text: Text to normalize.

        Returns:
            Normalized text string.
        """
        return " ".join(re.sub(r"[_-]", " ", text.lower()).split())

    def _looks_like_question(self, text: str) -> bool:
        """Heuristically detects whether an utterance is a question.

        Args:
            text: Input text.

        Returns:
            True if the text appears to be a question, else False.
        """
        stripped = text.strip()
        if not stripped:
            return False

        if stripped.endswith("?"):
            return True

        if stripped.endswith((".", "!")):
            return False

        normalized = self._normalize_text(stripped)
        return bool(
            re.match(
                (
                    r"^(who|what|when|where|why|how|which|"
                    r"is|are|was|were|do|does|did|can|could|"
                    r"would|will|should|have|has|had)\b"
                ),
                normalized,
            )
        )

    def _get_target_slot_values(self, slot: str) -> list[str]:
        """Returns normalized values for a slot across target items.

        Args:
            slot: Slot name.

        Returns:
            List of normalized slot values.
        """
        normalized_values: list[str] = []
        for item in self.information_need.target_items:
            value = item.get_property(slot)
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            normalized_values.extend(
                self._normalize_text(str(entry)) for entry in values
            )
        return normalized_values

    def _update_information_need_state_from_utterance(
        self, utterance: Utterance
    ) -> None:
        """Updates information-need state from a raw agent utterance.

        Args:
            utterance: Agent utterance.
        """
        raw = getattr(utterance, "text", "")
        text = self._normalize_text(raw)

        if not text:
            return

        is_question = self._looks_like_question(raw)

        for slot in self.information_need.request_states:
            normalized_slot = self._normalize_text(slot)
            target_values = self._get_target_slot_values(slot)
            mentions_slot_or_value = normalized_slot in text or any(
                value in text for value in target_values
            )

            if not mentions_slot_or_value:
                continue

            if is_question:
                self.information_need.mark_request_attempted(slot)
            else:
                self.information_need.mark_request_complete(slot, raw.strip())

        for slot, value in self.information_need.constraints.items():
            normalized_slot = self._normalize_text(slot)
            values = value if isinstance(value, list) else [value]

            normalized_values = [self._normalize_text(str(v)) for v in values]

            if any(v in text for v in normalized_values):
                if is_question:
                    self.information_need.mark_constraint_attempted(slot)
                else:
                    self.information_need.mark_constraint_complete(slot)
            elif normalized_slot in text:
                self.information_need.mark_constraint_attempted(slot)

    @abstractmethod
    def _generate_response(self, agent_utterance: Utterance) -> Utterance:
        """Generates response to the agent utterance.

        Args:
            agent_utterance: Agent utterance.

        Raises:
            NotImplementedError: If not implemented in derived class.

        Returns:
            User utterance.
        """
        raise NotImplementedError

    def receive_utterance(self, utterance: Utterance) -> None:
        """Gets called every time there is a new agent utterance.

        Args:
            utterance: Agent utterance.
        """
        response = self._generate_response(utterance)
        if not isinstance(response, AnnotatedUtterance):
            response = AnnotatedUtterance.from_utterance(response)
        self._dialogue_connector.register_user_utterance(response)
