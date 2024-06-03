"""Interface for dialogue state tracking."""

from typing import List

from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.participant import DialogueParticipant

from usersimcrs.dialogue_management.dialogue_state import DialogueState


class DialogueStateTracker:
    def __init__(self) -> None:
        """Initializes the dialogue state tracker."""
        self._dialogue_state = DialogueState()

    def get_current_state(self) -> DialogueState:
        """Returns the current dialogue state.

        Returns:
            DialogueState.
        """
        return self._dialogue_state

    def update_state(
        self,
        dialogue_acts: List[DialogueAct],
        participant: DialogueParticipant,
    ) -> None:
        """Updates the dialogue state based on the dialogue acts.

        Args:
            dialogue_acts: Dialogue acts.
            participant: Dialogue participant.
        """
        if participant == DialogueParticipant.USER:
            self._dialogue_state.user_dialogue_acts.append(dialogue_acts)
        else:
            self._dialogue_state.agent_dialogue_acts.append(dialogue_acts)

        self.update_belief_state(dialogue_acts)
        self._dialogue_state.turn_count += 1

    def update_belief_state(self, dialogue_acts: List[DialogueAct]) -> None:
        """Updates the belief state based on the dialogue acts.

        Args:
            dialogue_acts: Dialogue acts.
        """
        for dialogue_act in dialogue_acts:
            for annotation in dialogue_act.annotations:
                if annotation.value:
                    self._dialogue_state.belief_state[annotation.slot].append(
                        annotation.value
                    )
                else:
                    self._dialogue_state.belief_state[annotation.slot] = []

    def reset_state(self) -> None:
        """Resets the dialogue state."""
        self._dialogue_state = DialogueState()
