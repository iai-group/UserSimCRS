"""Wrapper for CRS agents available in iEvaLM fork.

The models are served via Flask API, the wrapper is responsible for sending
and receiving messages from an agent via the API.

The code for the agents is available at:
https://github.com/NoB0/iEvaLM-CRS
"""

from typing import Optional

import requests

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.intent import Intent
from dialoguekit.core.utterance import Utterance
from dialoguekit.participant import Agent
from dialoguekit.participant.agent import AgentType

DEFAULT_IEVALM_URI = "http://127.0.0.1:5005/"


class iEvaLMAgent(Agent):
    def __init__(
        self,
        id: str,
        agent_type: AgentType = AgentType.BOT,
        stop_intent: Intent = Intent("EXIT"),
        uri: str = DEFAULT_IEVALM_URI,
        crs_model: Optional[str] = None,
    ) -> None:
        """Initializes iEvaLM agent.

        Args:
            id: Agent ID.
            agent_type: Agent type. Defaults to AgentType.BOT.
            stop_intent: Label of the exit intent. Defaults to "EXIT".
            uri: URI of the iEvaLM agent. Defaults to DEFAULT_IEVALM_URI.
            crs_model: CRS model served by iEvaLM. Defaults to None.
        """
        super().__init__(id=id, agent_type=agent_type, stop_intent=stop_intent)
        self._uri = uri
        self._crs_model = crs_model

    def welcome(self) -> None:
        """Sends the agent's welcome message."""
        welcome_message = (
            f"Hello! I am {self._crs_model}. How can I help you?"
            if self._crs_model
            else "Hello! How can I help you?"
        )
        response = AnnotatedUtterance(
            text=welcome_message,
            participant=self._type,
        )
        self._dialogue_connector.register_agent_utterance(response)

    def goodbye(self) -> None:
        """Sends the agent's goodbye message."""
        goodbye_message = "Goodbye!"
        response = AnnotatedUtterance(
            text=goodbye_message,
            participant=self._type,
            dialogue_acts=[DialogueAct(self.stop_intent)],
        )
        self._dialogue_connector.register_agent_utterance(response)

    def receive_utterance(self, utterance: Utterance) -> None:
        """Responds to the other participant with an utterance.

        Args:
            utterance: The other participant's utterance.
        """
        context = []
        # Models expect the first utterance to be from the user. The agent
        # utterances before the user utterance are skipped.
        skip_agent_utterance = True
        for utterance in self._dialogue_connector.dialogue_history.utterances:
            speaker = utterance.participant
            if skip_agent_utterance and speaker == self._type:
                continue
            skip_agent_utterance = False
            context.append(utterance.text)

        r = requests.post(
            self._uri, json={"context": context, "message": utterance.text}
        )
        response = AnnotatedUtterance(
            text=r.text,
            participant=self._type,
        )
        self._dialogue_connector.register_agent_utterance(response)
