"""Define the prompt for stopping the conversation."""

from usersimcrs.core.information_need import InformationNeed
from usersimcrs.simulator.llm.prompt.prompt import Prompt
from usersimcrs.user_modeling.persona import Persona

DEFAULT_STOP_DEFINITION = (
    "As a USER interacting with an ASSISTANT to receive a recommendation, "
    "analyze the conversation history to determine if it is progressing "
    "productively. If the conversation has been stuck in a loop with repeated "
    "misunderstandings across multiple turns, return 'FALSE' to indicate the "
    "conversation should be terminated. Otherwise, return 'TRUE' to indicate "
    "that the conversation should continue. Only return 'TRUE' or 'FALSE' "
    "without any additional information."
)


class StopPrompt(Prompt):
    def __init__(
        self,
        information_need: InformationNeed,
        item_type: str,
        prompt_definition: str = DEFAULT_STOP_DEFINITION,
        persona: Persona = None,
    ) -> None:
        """Initializes the prompt.

        Args:
            information_need: The information need of the user.
            item_type: The type of the item to be recommended.
            prompt_definition: The definition of the task to be performed.
              Defaults to DEFAULT_STOP_DEFINITION.
            persona: The persona of the user. Defaults to None.
        """
        super().__init__(
            information_need, item_type, prompt_definition, persona
        )

    @property
    def prompt_text(self) -> str:
        """Prompt for the user simulator."""
        return (
            self._initial_prompt
            + "\n"
            + self._prompt_context
            + "\n"
            + "CONTINUE: "
        )

    def build_new_prompt(self) -> str:
        """Builds the initial prompt without any context.

        Returns:
            Initial prompt with task definition, requirements, and persona.
        """
        initial_prompt = self.prompt_definition

        if self.persona:
            initial_prompt += (
                " Take into account your PERSONA when deciding to stop the "
                "conversation.\n"
            )
            stringified_characteristics = ", ".join(
                [
                    f"{key}={value}"
                    for key, value in self.persona.characteristics.items()
                ]
            )
            initial_prompt += f"PERSONA: {stringified_characteristics}\n"

        initial_prompt += "\nHISTORY:\n"
        return initial_prompt
