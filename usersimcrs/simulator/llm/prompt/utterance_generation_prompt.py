"""Define the prompt for the simulator.

The structure of the prompt is inspired by the work of Terragni et al. It
includes the task description, the information need, and optionally a persona
and an example of a conversation. Unlike the original work, we consider a zero-
shot setting, i.e., the prompt does not include any examples of conversations.

Reference: Terragni, S., et al. (2023). "In-Context Learning User Simulators
for Task-Oriented Dialog Systems", arXiv 2306.00774.
"""

from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.participant import DialogueParticipant
from usersimcrs.core.information_need import InformationNeed
from usersimcrs.simulator.llm.prompt.prompt import Prompt
from usersimcrs.user_modeling.persona import Persona

DEFAULT_TASK_DEFINITION = (
    "You are a USER discussing with an ASSISTANT. Given the conversation "
    "history, you need to generate the next USER message in the most natural "
    "way possible. The conversation is about getting a recommendation "
    "according to the REQUIREMENTS. You must fulfill all REQUIREMENTS as the "
    "conversation progresses (you don't need to fulfill them all at once). "
    "After getting all the necessary information, you can terminate the "
    "conversation by sending '\\end'. You may also terminate the conversation "
    "is not going anywhere or the ASSISTANT is not helpful by sending "
    "'\\giveup'. "
)


class UtteranceGenerationPrompt(Prompt):
    def __init__(
        self,
        information_need: InformationNeed,
        item_type: str,
        prompt_definition: str = DEFAULT_TASK_DEFINITION,
        persona: Persona = None,
    ) -> None:
        """Initializes the prompt.

        Args:
            information_need: The information need of the user.
            item_type: The type of the item to be recommended.
            prompt_definition: The definition of the task to be performed.
              Defaults to DEFAULT_TASK_DEFINITION.
            persona: The persona of the user. Defaults to None.
        """
        super().__init__(
            information_need, item_type, prompt_definition, persona
        )

    def build_new_prompt(self) -> str:
        """Builds the initial prompt without any context.

        Returns:
            Initial prompt with task definition, requirements, and persona.
        """
        initial_prompt = self.prompt_definition

        if self.persona:
            initial_prompt += (
                " Adapt your responses considering your PERSONA.\n"
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
                f"{key.lower()}={value}"
                for key, value in self.information_need.constraints.items()
            ]
        )
        requestable_slot = ", ".join(
            [k.lower() for k in self.information_need.requested_slots.keys()]
        )
        initial_prompt += (
            f"\nREQUIREMENTS: You are looking for a {self.item_type} with the "
            f"following characteristics: {stringified_constraints}. Once you "
            f"find a suitable {self.item_type}, make sure to get the following "
            f"information: {requestable_slot}.\nHISTORY:\n"
        )
        return initial_prompt

    def update_prompt_context(
        self, utterance: Utterance, participant: DialogueParticipant
    ) -> None:
        """Updates the context provided in the prompt.

        Args:
            utterance: Utterance to be added to the prompt.
            participant: Participant of the conversation.
        """
        super().update_prompt_context(utterance, participant)
        if participant == DialogueParticipant.AGENT:
            self._prompt_context += "USER: "
