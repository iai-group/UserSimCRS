"""Representation of the dialogue state.

The dialogue state includes the turn count, a list of dialogue acts per turn
for both the agent and the user, and the belief state. The belief state is a
dictionary that maps slot names to slot values.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, List

from dialoguekit.core.dialogue_act import DialogueAct


@dataclass
class DialogueState:
    """Dialogue state.

    Attributes:
        turn_count: Turn count.
        agent_dacts: List of dialogue acts per turn for the agent.
        user_dacts: List of dialogue acts per turn for the user.
        belief_state: Belief state.
    """

    turn_count: int = 0
    agent_dacts: List[List[DialogueAct]] = field(default_factory=list)
    user_dacts: List[List[DialogueAct]] = field(default_factory=list)
    belief_state: DefaultDict[str, str] = field(
        default_factory=lambda: defaultdict(str)
    )
