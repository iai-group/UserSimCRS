"""Interface to represent an agenda.

The agenda is a stack of dialogue acts that the user wants to perform to fulfill
their information need. The representation is based on the description in:
Agenda-Based User Simulation for Bootstrapping a POMDP Dialogue System,
Schatzmann et al., 2007.
"""

from collections import deque
from typing import List

from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.intent import Intent
from dialoguekit.core.slot_value_annotation import SlotValueAnnotation
from usersimcrs.core.information_need import InformationNeed


class Agenda:
    def __init__(
        self,
        information_need: InformationNeed,
        inform_intent: Intent,
        request_intent: Intent,
        stop_intent: Intent,
        start_intent: Intent = None,
    ) -> None:
        """Initializes the agenda.

        Args:
            information_need: Information need.
            inform_intent: Inform intent.
            request_intent: Request intent.
            stop_intent: Stop intent.
            start_intent: Start intent. Defaults to None.
        """
        self._dialogue_acts_stack = deque()
        self.inform_intent = inform_intent
        self.request_intent = request_intent

        if start_intent is not None:
            self._dialogue_acts_stack.append(DialogueAct(start_intent))

        # All constraints are converted to inform dialogue acts
        for slot, value in information_need.constraints.items():
            if isinstance(value, list):
                annotations = [SlotValueAnnotation(slot, v) for v in value]
            else:
                annotations = [SlotValueAnnotation(slot, value)]
            self._dialogue_acts_stack.append(
                DialogueAct(inform_intent, annotations)
            )

        # All requests are converted to request dialogue acts
        for slot in information_need.get_requestable_slots():
            self._dialogue_acts_stack.append(
                DialogueAct(request_intent, [SlotValueAnnotation(slot)])
            )

        # Finish with a stop dialogue act
        self._dialogue_acts_stack.append(DialogueAct(stop_intent))

    @property
    def stack(self) -> List[DialogueAct]:
        """Returns the dialogue acts stack."""
        return self._dialogue_acts_stack

    def get_next_dialogue_acts(self, n: int) -> List[DialogueAct]:
        """Returns the next n dialogue acts from the stack.

        Args:
            n: Number of dialogue acts to return.

        Returns:
            List of dialogue acts.
        """
        return list([self._dialogue_acts_stack.popleft() for _ in range(n)])

    def push_dialogue_act(self, dialogue_act: DialogueAct) -> None:
        """Pushes a dialogue act onto the stack.

        Args:
            dialogue_act: Dialogue act.
        """
        self._dialogue_acts_stack.appendleft(dialogue_act)

    def push_dialogue_acts(self, dialogue_acts: List[DialogueAct]) -> None:
        """Pushes dialogue acts onto the stack.

        Args:
            dialogue_acts: Dialogue acts.
        """
        for dialogue_act in dialogue_acts:
            self.push_dialogue_act(dialogue_act)

    def clean_agenda(self, information_need: InformationNeed) -> None:
        """Cleans the agenda.

        Removes duplicate dialogue acts, null dialogue acts, and requests for
        already informed slots.

        Args:
            information_need: Information need.
        """
        new_stack = deque()
        informed_slots = information_need.get_requestable_slots()
        for dialogue_act in self._dialogue_acts_stack:
            if dialogue_act is None:
                continue
            if dialogue_act.intent == self.request_intent and any(
                annotation.slot in informed_slots
                for annotation in dialogue_act.annotations
            ):
                continue
            if dialogue_act not in new_stack:
                new_stack.append(dialogue_act)

        self._dialogue_acts_stack = new_stack
