"""User simulator leveraging a large language model to generate responses.

The generation of responses is based on two prompts. The first one establishes
if the conversation should continue or not. The second one is used to generate
the user response.
"""

from dialoguekit.core.utterance import Utterance
from dialoguekit.participant import DialogueParticipant
from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.simulator.llm.interfaces.llm_interface import LLMInterface
from usersimcrs.simulator.llm.prompt.stop_prompt import (
    DEFAULT_STOP_DEFINITION,
    StopPrompt,
)
from usersimcrs.simulator.llm.prompt.utterance_generation_prompt import (
    DEFAULT_TASK_DEFINITION,
    UtteranceGenerationPrompt,
)
from usersimcrs.simulator.user_simulator import UserSimulator
from usersimcrs.user_modeling.persona import Persona


class DualPromptUserSimulator(UserSimulator):
    def __init__(
        self,
        id: str,
        domain: SimulationDomain,
        item_collection: ItemCollection,
        llm_interface: LLMInterface,
        item_type: str,
        task_definition: str = DEFAULT_TASK_DEFINITION,
        stop_definition: str = DEFAULT_STOP_DEFINITION,
        persona: Persona = None,
    ) -> None:
        """Initializes the user simulator.

        Args:
            id: User simulator ID.
            llm_interface: Interface to the large language model.
            item_type: Type of the item to be recommended. Defaults to None.
            task_definition: Definition of the task to be performed.
              Defaults to DEFAULT_TASK_DEFINITION.
            stop_definition: Definition of the stop task. Defaults to
              DEFAULT_STOP_DEFINITION.
            persona: Persona of the user. Defaults to None.
        """
        super().__init__(id, domain, item_collection)
        self.llm_interface = llm_interface
        self.generation_prompt = UtteranceGenerationPrompt(
            self.information_need, item_type, task_definition, persona
        )
        self.stop_prompt = StopPrompt(
            self.information_need, item_type, stop_definition, persona
        )

    def _generate_response(self, agent_utterance: Utterance) -> Utterance:
        """Generates response to the agent utterance.

        Args:
            agent_utterance: Agent utterance.

        Returns:
            User utterance.
        """
        self.generation_prompt.update_prompt_context(
            agent_utterance, DialogueParticipant.AGENT
        )
        self.stop_prompt.update_prompt_context(
            agent_utterance, DialogueParticipant.AGENT
        )

        # Check if the conversation should continue
        b_continue = self.llm_interface.get_llm_response(
            self.stop_prompt.prompt_text
        )
        if b_continue.strip().lower() == "false":
            user_utterance = Utterance("\\end", DialogueParticipant.USER)
        else:
            user_utterance = self.llm_interface.generate_response(
                self.generation_prompt
            )
        self.generation_prompt.update_prompt_context(
            user_utterance, DialogueParticipant.USER
        )
        return user_utterance
