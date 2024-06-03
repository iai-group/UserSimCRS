"""Representation of the dialogue state.

The dialogue state includes the number of utterance, a list of dialogue acts per utterance for both
the agent and the user, and the belief state. The belief state is a dictionary
that holds for each slot (key) the values provided by the user.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, List

from dialoguekit.core.dialogue_act import DialogueAct


@dataclass
class DialogueState:
    """Dialogue state.

    Attributes:
        utterance_count: Utterance count.
        agent_dialogue_acts: List of dialogue acts per turn for the agent.
        user_dialogue_acts: List of dialogue acts per turn for the user.
        belief_state: Belief state.
    """

    utterance_count: int = 0
    agent_dialogue_acts: List[List[DialogueAct]] = field(default_factory=list)
    user_dialogue_acts: List[List[DialogueAct]] = field(default_factory=list)
    belief_state: DefaultDict[str, List[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
