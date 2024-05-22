"""Simulation platform to connect simulator and agent."""

import logging
import sys
from typing import Any, Dict, Type

import requests
from dialoguekit.connector import DialogueConnector
from dialoguekit.participant import Agent
from dialoguekit.platforms import Platform

from usersimcrs.simulator.user_simulator import UserSimulator


class SimulationPlatform(Platform):
    def __init__(
        self, agent_class: Type[Agent], agent_config: Dict[str, Any] = {}
    ) -> None:
        """Initializes the simulation platform.

        Args:
            agent_class: Agent class.
            agent_config: Configuration of the agent. Defaults to empty
              dictionary.
        """
        super().__init__(agent_class)
        self._agent_class = agent_class
        self._agent_config = agent_config

    def start(self) -> None:
        """Starts the simulation platform.

        It creates the agent and user simulator.

        Raises:
            RuntimeError: If the connection to the agent is refused.
            ValueError: If the agent URI is not specified in the agent
              configuration.
        """
        try:
            agent_uri = self._agent_config["uri"]
            response = requests.get(agent_uri, timeout=60)
            assert response.status_code == 200
            self.agent = self._agent_class(**self._agent_config)
        except requests.exceptions.RequestException:
            raise RuntimeError(
                f"Connection refused to {agent_uri}. Please check that "
                "the conversational agent is running at this address. See the "
                "full traceback above."
            )
        except KeyError:
            raise ValueError(
                "The agent URI is not specified in the agent configuration."
            )

    def connect(
        self,
        user_id: str,
        simulator_class: Type[UserSimulator],
        simulator_config: Dict[str, Any] = {},
    ) -> None:
        """Connects a user simulator to an agent.

        Args:
            user_id: User ID.
            simulator_class: User simulator class.
            simulator_config: Configuration of the user simulator. Defaults to
              empty dictionary.

        Raises:
            Exception: If an error occurs during the dialogue.
        """
        self._active_users[user_id] = simulator_class(
            user_id, **simulator_config
        )
        dialogue_connector = DialogueConnector(
            agent=self.agent,
            user=self._active_users[user_id],
            platform=self,
        )

        try:
            dialogue_connector.start()
        except Exception as e:
            tb = sys.exc_info()
            dialogue_connector._dialogue_history._metadata.update(
                {
                    "error": {
                        "error_type": type(e).__name__,
                        "trace": str(e.with_traceback(tb[2])),
                    }
                }
            )
            return

    def display_agent_utterance(self, user_id: str, utterance: str) -> None:
        """Displays an agent utterance.

        Args:
            user_id: Agent ID.
            utterance: An instance of Utterance.
        """
        logging.debug(f"Agent {user_id}: {utterance.text}")

    def display_user_utterance(self, user_id: str, utterance: str) -> None:
        """Displays a user utterance.

        Args:
            user_id: User ID.
            utterance: An instance of Utterance.
        """
        logging.debug(f"User {user_id}: {utterance.text}")
