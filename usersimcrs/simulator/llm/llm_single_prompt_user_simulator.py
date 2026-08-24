"""User simulator leveraging a large language model to generate responses.

The responses are generated via a single prompt template with a large language
model.
"""

from typing import Optional

from dialoguekit.core.utterance import Utterance
from dialoguekit.participant import DialogueParticipant
from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.information_need_management.information_need_tracker import (
    InformationNeedTracker,
)
from usersimcrs.items.item_collection import ItemCollection
from usersimcrs.llm_interfaces.llm_interface import LLMInterface
from usersimcrs.simulator.llm.prompt.utterance_generation_prompt import (
    DEFAULT_TASK_DEFINITION,
    UtteranceGenerationPrompt,
)
from usersimcrs.simulator.user_simulator import UserSimulator
from usersimcrs.user_modeling.preference_model import PreferenceModel
from usersimcrs.user_modeling.persona import Persona


class LLMSinglePromptUserSimulator(UserSimulator):
    def __init__(
        self,
        id: str,
        domain: SimulationDomain,
        item_collection: ItemCollection,
        llm_interface: LLMInterface,
        item_type: str,
        information_need_tracker: InformationNeedTracker,
        task_definition: str = DEFAULT_TASK_DEFINITION,
        persona: Optional[Persona] = None,
        preference_model: Optional[PreferenceModel] = None,
    ) -> None:
        """Initializes the user simulator.

        Args:
            id: User simulator ID.
            llm_interface: Interface to the large language model.
            item_type: Type of the item to be recommended. Defaults to None.
            information_need_tracker: Tracker for the information need.
            task_definition: Definition of the task to be performed.
              Defaults to DEFAULT_TASK_DEFINITION.
            persona: Persona of the user. Defaults to None.
            preference_model: Preference model. Defaults to None.
        """
        super().__init__(
            id,
            domain,
            item_collection,
            information_need_tracker,
            preference_model,
        )
        self.llm_interface = llm_interface
        self.prompt = UtteranceGenerationPrompt(
            self.information_need_tracker.get_information_need(),
            item_type,
            task_definition,
            persona,
            preference_model,
        )

    def _generate_response(self, agent_utterance: Utterance) -> Utterance:
        """Generates response to the agent utterance.

        Args:
            agent_utterance: Agent utterance.

        Returns:
            User utterance.
        """
        self.information_need_tracker.apply_updates_to_information_need(
            self.information_need_tracker.update_from_utterance(agent_utterance)
        )
        self.prompt.update_prompt_context(
            agent_utterance, DialogueParticipant.AGENT
        )
        user_utterance = self.llm_interface.generate_utterance(self.prompt)
        self.prompt.update_prompt_context(
            user_utterance, DialogueParticipant.USER
        )
        self.information_need_tracker.apply_updates_to_information_need(
            self.information_need_tracker.update_from_utterance(user_utterance)
        )
        return user_utterance
