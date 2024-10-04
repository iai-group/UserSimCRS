"""Interface for prompt."""

from abc import ABC, abstractmethod

from dialoguekit.core.utterance import Utterance
from dialoguekit.participant.participant import DialogueParticipant
from usersimcrs.core.information_need import InformationNeed
from usersimcrs.user_modeling.persona import Persona


class Prompt(ABC):
    def __init__(
        self,
        information_need: InformationNeed,
        item_type: str,
        prompt_definition: str,
        persona: Persona = None,
    ) -> None:
        """Initializes the prompt.

        Args:
            information_need: The information need of the user.
            item_type: The type of the item to be recommended.
            prompt_definition: The definition of the task to be performed.
            persona: The persona of the user. Defaults to None.
        """
        self.information_need = information_need
        self.item_type = item_type
        self.prompt_definition = prompt_definition
        self.persona = persona
        self._initial_prompt = self.build_new_prompt()
        self._prompt_context = ""

    @property
    def prompt_text(self) -> str:
        """Prompt for the user simulator."""
        return self._initial_prompt + "\n" + self._prompt_context

    @abstractmethod
    def build_new_prompt(self, **kwargs) -> str:
        """Builds the initial prompt without any context.

        Raises:
            NotImplementedError: If the method is not implemented in subclasses.

        Returns:
            Initial prompt.
        """
        raise NotImplementedError

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

        self._prompt_context += f"{role}: {utterance.text}\n"
