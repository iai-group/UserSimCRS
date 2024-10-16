"""Define the prompt for stopping the conversation."""

from usersimcrs.core.information_need import InformationNeed
from usersimcrs.simulator.llm.prompt.prompt import Prompt
from usersimcrs.user_modeling.persona import Persona

DEFAULT_STOP_DEFINITION = (
    "You are a USER discussing with an ASSISTANT to get a recommendation "
    "meeting your REQUIREMENTS. Given the conversation history, you need to "
    "decide whether to continue the conversation or not. Detect if the "
    "conversation is not progressing towards your goal or if the ASSISTANT is "
    "not helpful. In such cases, you should terminate the conversation by "
    "returning TRUE. Otherwise, return FALSE."
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
            f"following characteristics: {stringified_constraints} and want to "
            f"know the following information about it: {requestable_slot}.\n"
            "HISTORY:\n"
        )
        return initial_prompt
