from dialoguekit.participant.agent import Agent


class SampleAgent(Agent):
    """Sample recommender agent."""

    def __init__(self, agent_id: str):
        """Initializes agent.

        Args:
            agent_id: Agent id.
        """
        super().__init__(agent_id)