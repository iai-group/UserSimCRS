"""Define the prompt for the simulator.

The structure of the prompt is inspired by the work of Terragni et al. It
includes the task description, the information need, and optionally a persona
and an example of a conversation. Unlike the original work, we consider a zero-
shot setting, i.e., the prompt does not include any examples of conversations.

Reference: Terragni, S., et al. (2023). "In-Context Learning User Simulators
for Task-Oriented Dialog Systems", arXiv 2306.00774.
"""

from dialoguekit.core import Utterance
from dialoguekit.participant import DialogueParticipant

from usersimcrs.core.information_need import InformationNeed
from usersimcrs.user_modeling.persona import Persona

DEFAULT_TASK_DEFINITION = """Complete the conversation as a CUSTOMER
 discussing with an ASSISTANT. The conversation is about getting a
 recommendation according to the REQUIREMENTS. You must fulfill all
 REQUIREMENTS."""


class Prompt:
    def __init__(
        self,
        requirements: InformationNeed,
        item_type: str,
        task_definition: str = DEFAULT_TASK_DEFINITION,
        persona: Persona = None,
    ) -> None:
        """Initializes the prompt.

        Args:
            requirements: The information need of the user.
            item_type: The type of the item to be recommended.
            task_definition: The definition of the task to be performed.
              Defaults to DEFAULT_TASK_DEFINITION.
            persona: The persona of the user. Defaults to None.
        """
        self.requirements = requirements
        self.item_type = item_type
        self.task_definition = task_definition
        self.persona = persona
        self._initial_prompt = self.build_new_prompt()
        self._prompt_context = ""

    @property
    def prompt_text(self) -> str:
        """Prompt for the user simulator."""
        return self._initial_prompt + "\n" + self._prompt_context

    def build_new_prompt(self) -> str:
        """Builds the initial prompt without any context.

        Returns:
            Initial prompt with task definition, requirements, and persona.
        """
        initial_prompt = self.task_definition

        if self.persona:
            initial_prompt += (
                "Adapt your responses considering the your PERSONA.\n"
            )
            stringified_characteristics = ", ".join(
                [
                    f"{key}={value}"
                    for key, value in self.persona.characteristics.items()
                ]
            )
            initial_prompt += f"PERSONA: {stringified_characteristics}\n"
        else:
            initial_prompt += (
                "Be precise with the REQUIREMENTS, clear and concise.\n"
            )

        stringified_constraints = ", ".join(
            [
                f"{key}={value}"
                for key, value in self.requirements.constraints.items()
            ]
        )
        initial_prompt += (
            f"REQUIREMENTS: You are looking for a {self.item_type} "
        )
        f"with the following characteristics: {stringified_constraints}. Once "
        f"you find a suitable {self.item_type}, make sure to get the following "
        f"information: {', '.join(self.requirements.requests.keys())}.\n"
        return initial_prompt

    def update_prompt_context(
        self, utterance: Utterance, participant: DialogueParticipant
    ) -> None:
        """Updates the context provided in the prompt.

        Args:
            utterance: Utterance to be added to the prompt.
            participant: Participant of the conversation.
        """
        role = (
            "ASSISTANT" if participant == DialogueParticipant.AGENT else "USER"
        )

        self._prompt_context += f" {role}: {utterance.text}\n"
        if participant == DialogueParticipant.AGENT:
            self._prompt_context += "USER: "
