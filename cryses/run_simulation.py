"""Console application for running simulation."""

from dialoguekit.agent.agent import Agent
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.ontology import Ontology
from dialoguekit.core.recsys.item_collection import ItemCollection
from dialoguekit.core.recsys.ratings import Ratings
from dialoguekit.manager.dialogue_manager import DialogueManager

from cryses.simulator.user_simulator import UserSimulator
from cryses.sample_agent.sample_agent import SampleAgent
from cryses.simulator.agenda_based_simulator import AgendaBasedSimulator


def simulate_conversation(agent: Agent, user_simulator: UserSimulator) -> Dialogue:
    """Simulates a single conversation and returns the resulting dialogue.

    Args:
        agent: An agent.
        user_simulator: A user simulator.
    """
    platform = None  # TODO: Add simulator platform
    dm = DialogueManager(agent, user_simulator, platform)
    dm.start()
    dm.close()
    return dm.dialogue_history


if __name__ == "__main__":
    agent = SampleAgent(agent_id="sample_agent")

    # TODO Load settings from command line arguments or config file.
    ontology_yaml = "../dialoguekit/tests/data/ontology.yaml"
    items_csv_file = "../dialoguekit/tests/data/movielens-20m-sample/movies.csv"
    ratings_csv_file = "tests/data/movielens-20m-sample/ratings.csv"

    ontology = Ontology(ontology_yaml)

    item_collection = ItemCollection()
    item_collection.load_items_csv(items_csv_file, ["ID", "NAME", "genres"])

    ratings = Ratings(item_collection)
    ratings.load_ratings_csv(ratings_csv_file)

    # TODO: initialization of the simulator with NLU, NLG, etc.
    preference_model = None
    interaction_model = None
    nlu = None
    nlg = None
    simulator = AgendaBasedSimulator(
        preference_model,
        interaction_model,
        nlu,
        nlg,
        ontology,
        item_collection,
        ratings,
    )
    simulate_conversation(agent, simulator)
