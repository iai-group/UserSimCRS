"""Console application for running simulation."""

from dialoguekit.agent.agent import Agent
from dialoguekit.dialogue.dialogue import Dialogue
from dialoguekit.manager.dialogue_manager import DialogueManager
from cryses.simulator.simulated_user import SimulatedUser

from cryses.sample_agent.sample_agent import SampleAgent
from cryses.simulator.agenda_based_simulator import AgendaBasedSimulator


def simulate_conversation(agent: Agent, simulated_user: SimulatedUser) -> Dialogue:
    """Simulates a single conversation and returns the resulting dialogue.


    """
    platform = None   # TODO: Add simulator platform
    dm = DialogueManager(agent, simulated_user, platform)
    dm.start()
    dm.close()
    return dm.dialogue_history


if __name__ == "__main__":
    agent = SampleAgent(agent_id="sample_agent")
    # TODO: initialization of the simulator with NLU, NLG, etc.
    simulator = AgendaBasedSimulator()
    simulate_conversation(agent, simulator)
