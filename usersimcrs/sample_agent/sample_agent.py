from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.intent import Intent
from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.agent import Agent
from dialoguekit.participant.participant import DialogueParticipant


class SampleAgent(Agent):
    """Sample recommender agent."""

    def __init__(self, agent_id: str):
        """Initializes agent.

        Args:
            agent_id: Agent id.
        """
        super().__init__(agent_id)

    def welcome(self) -> None:
        utterance = AnnotatedUtterance(
            "Hello, I'm SampleAgent.",
            participant=DialogueParticipant.AGENT,
        )
        self._dialogue_connector.register_agent_utterance(utterance)

    def goodbye(self) -> None:
        utterance = AnnotatedUtterance(
            "It was nice talking to you. Bye",
            intent=Intent("EXIT"),
            participant=DialogueParticipant.AGENT,
        )
        self._dialogue_connector.register_agent_utterance(utterance)

    def receive_utterance(self, utterance: Utterance) -> None:
        if utterance.text == "EXIT":
            self.goodbye()
            return

        response = AnnotatedUtterance(
            "This sample agent does not perform actions.",
            participant=DialogueParticipant.AGENT,
        )
        self._dialogue_connector.register_agent_utterance(response)
