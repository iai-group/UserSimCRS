"""User simulator leveraging a large language model to generate responses.

The responses are generated via a single prompt template with a large language
model. The prompt is inspired by the work of Terragni et al.

Reference: Terragni, S., et al. (2023). "In-Context Learning User Simulators
for Task-Oriented Dialog Systems", arXiv 2306.00774.
"""

from dialoguekit.core.utterance import Utterance

from usersimcrs.simulator.llm.interfaces.llm_interface import LLMInterface
from usersimcrs.simulator.user_simulator import UserSimulator


class SinglePromptUserSimulator(UserSimulator):
    def __init__(
        self,
        id: str,
        llm_interface: LLMInterface,
        initial_prompt: str,
    ) -> None:
        """Initializes the user simulator.

        Args:
            id: User simulator ID.
            llm_interface: Interface to the large language model.
            initial_prompt: Initial prompt to be used for generating responses.
            default_response: Default response to be used if the LLM fails to
              generate a response.
        """
        super().__init__(id)
        self.llm_interface = llm_interface
        self.prompt = initial_prompt

    def _generate_response(self, agent_utterance: Utterance) -> Utterance:
        """Generates response to the agent utterance.

        Args:
            agent_utterance: Agent utterance.

        Returns:
            User utterance.
        """
        self.update_prompt_context(agent_utterance, role="Agent")
        user_utterance = self.llm_interface.generate_response(self.prompt)
        self.update_prompt_context(user_utterance, role="User")
        return user_utterance

    def update_prompt_context(self, utterance: Utterance, role: str) -> None:
        """Updates the context provided in the prompt.

        Args:
            utterance: Utterance to be added to the prompt.
            role: Role of the participant, either "Agent" or "User".

        Raises:
            ValueError: If the role is not valid.
        """
        if role not in ["Agent", "User"]:
            raise ValueError(f"Invalid role: {role}")

        self.prompt += f" {role}: {utterance.text}\n"
        if role == "Agent":
            self.prompt += "User:"
