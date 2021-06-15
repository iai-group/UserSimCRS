"""Interaction model."""

# TODO: change intent types to Intent once that's been added to dialoguekit.core

import os
import yaml


class InteractionModel:
    """Represents an interaction model."""

    def __init__(self, config_file) -> None:
        """Initializes the interaction model."""
        # Load interaction model.
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with open(config_file) as yaml_file:
            self._config = yaml.load(yaml_file, Loader=yaml.FullLoader)

        # Initialize agenda.
        self._agenda = []
        # Keep track of the current user intent.
        self._current_intent = None

    def initialize_agenda(self) -> None:
        """Initializes the action agenda."""
        pass

    @property
    def agenda(self):
        return self._agenda

    @property
    def current_intent(self) -> str:
        return self._current_intent

    def is_agent_intent_elicit(self, agent_intent: str) -> bool:
        """Checks if the given agent intent is elicitation.

        Args:
            agent_intent: Agent's intent.

        Returns:
            True if it is an elicitation intent.
        """
        return agent_intent in self._config["agent_elicit_intents"]

    def is_agent_intent_set_retrieval(self, agent_intent: str) -> bool:
        """Checks if the given agent intent is set retrieval.

        Args:
            agent_intent: Agent's intent.

        Returns:
            True if it is a set retrieval intent.
        """
        return agent_intent in self._config["agent_set_retrieval"]

    def update_agenda(self, agent_intent: str) -> None:
        """Updates the agenda and determines the next user intent based on agent
        intent.

        If agent replies with an expected intent in response to the last user
        intent (based on the expected_responses mapping in the config file),
        then
            pops up the next user intent from the agenda;
            update the current intent;
        Otherwise:
            pushes a new intent (select a replacement intent);
            updates the current intent.

        Args:
            agent_intent: Agent's intent
        """
        pass

