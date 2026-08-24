"""User simulator abstract class."""

import json
from abc import ABC, abstractmethod
from typing import Optional

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.user import User, UserType

from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.information_need_management.information_need import (
    InformationNeed,
)
from usersimcrs.information_need_management.information_need_tracker import (
    InformationNeedTracker,
)
from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.user_modeling.preference_model import PreferenceModel


class UserSimulator(User, ABC):
    TERMINATION_COMMANDS = {
        "\\end",
        "\\giveup",
        "/end",
        "/exit",
        "end",
        "exit",
        "give up",
        "giveup",
        "stop",
    }

    def __init__(
        self,
        id: str,
        domain: SimulationDomain,
        item_collection: ItemCollection,
        information_need_tracker: InformationNeedTracker,
        preference_model: Optional[PreferenceModel] = None,
    ) -> None:
        """Initializes the user simulator.

        Args:
            id: Simulator ID.
            domain: Domain.
            item_collection: Item collection.
            information_need_tracker: Tracker for the information need.
            preference_model: Preference model. Defaults to None.
        """
        super().__init__(id, UserType.SIMULATOR)
        self._domain = domain
        self._item_collection = item_collection
        self.preference_model = preference_model
        self.information_need_tracker = information_need_tracker
        self._last_preference_summary = ""
        self._print_information_need()

    @property
    def information_need(self) -> InformationNeed:
        """Returns the current tracked information need."""
        return self.information_need_tracker.get_information_need()

    def _print_information_need(self) -> None:
        """Prints the information need once at simulator initialization."""
        print("INFORMATION NEED")
        print(json.dumps(self.information_need.to_dict(), indent=2))

    def _print_preferences_if_changed(self) -> None:
        """Prints preference summary when it changes."""
        if not self.preference_model:
            return

        preference_summary = self.preference_model.get_preference_summary()
        if preference_summary == self._last_preference_summary:
            return

        self._last_preference_summary = preference_summary
        print("PREFERENCES")
        print(preference_summary or "<empty>")

    def _is_terminal_utterance(self, utterance: Utterance) -> bool:
        """Returns whether the utterance explicitly terminates the dialogue."""
        normalized_text = utterance.text.strip().strip("\"'").lower()
        return normalized_text in self.TERMINATION_COMMANDS

    def _register_user_utterance_and_close(
        self, response: AnnotatedUtterance
    ) -> None:
        """Registers a terminal user utterance without sending it to the
        agent."""
        self._dialogue_connector.dialogue_history.add_utterance(response)
        self._dialogue_connector.get_platform().display_user_utterance(
            response, self.id
        )
        self._dialogue_connector.close()

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

        if self._is_terminal_utterance(response):
            self._register_user_utterance_and_close(response)
            return

        if self.preference_model:
            self.preference_model.update_from_utterance(response)
            self._print_preferences_if_changed()
        self._dialogue_connector.register_user_utterance(response)
