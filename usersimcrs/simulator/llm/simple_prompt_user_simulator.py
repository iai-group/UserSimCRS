"""User simulator leveraging a large language model to generate responses.

The responses are generated via a single prompt template with a large language
model.
"""

from dialoguekit.core.utterance import Utterance
from dialoguekit.participant import DialogueParticipant

from usersimcrs.simulator.llm.interfaces.llm_interface import LLMInterface
from usersimcrs.simulator.llm.prompt import DEFAULT_TASK_DEFINITION, Prompt
from usersimcrs.simulator.user_simulator import UserSimulator
from usersimcrs.user_modeling.persona import Persona


class SinglePromptUserSimulator(UserSimulator):
    def __init__(
        self,
        id: str,
        llm_interface: LLMInterface,
        item_type: str,
        task_definition: str = DEFAULT_TASK_DEFINITION,
        persona: Persona = None,
    ) -> None:
        """Initializes the user simulator.

        Args:
            id: User simulator ID.
            llm_interface: Interface to the large language model.
            item_type: Type of the item to be recommended. Defaults to None.
            task_definition: Definition of the task to be performed.
              Defaults to DEFAULT_TASK_DEFINITION.
            persona: Persona of the user. Defaults to None.
        """
        super().__init__(id)
        self.llm_interface = llm_interface
        self.prompt = Prompt(
            self.information_need, item_type, task_definition, persona
        )

    def _generate_response(self, agent_utterance: Utterance) -> Utterance:
        """Generates response to the agent utterance.

        Args:
            agent_utterance: Agent utterance.

        Returns:
            User utterance.
        """
        self.prompt.update_prompt_context(
            agent_utterance, DialogueParticipant.AGENT
        )
        user_utterance = self.llm_interface.generate_response(self.prompt)
        self.prompt.update_prompt_context(
            user_utterance, DialogueParticipant.USER
        )
        return user_utterance
