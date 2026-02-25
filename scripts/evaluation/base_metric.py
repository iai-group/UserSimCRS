"""Abstract base class for dialogue evaluation metrics."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List
from dialoguekit.core.dialogue import Dialogue


class Metric(ABC):
    def __init__(self, name: str) -> None:
        """Initializes the metric.

        Args:
            name: Metric name.
        """
        self.name = name

    @abstractmethod
    def evaluate_dialogue(self, dialogue: Dialogue, **kwargs: Any) -> float:
        """Computes the metric for a single dialogue.

        Args:
            dialogue: Single dialogue to score.
            **kwargs: Additional arguments specific to the metric.

        Raises:
            NotImplementedError: When not implemented by a subclass.

        Returns:
            Score for the dialogue.
        """
        raise NotImplementedError()

    def evaluate_dialogues(
        self, dialogues: List[Dialogue], **kwargs: Any
    ) -> Dict[str, float]:
        """Computes the metric for every dialogue in a given list.

        Args:
            dialogues: Dialogues.
            **kwargs: Additional arguments specific to the metric.

        Returns:
            Dictionary with result per dialogue.
        """
        return {
            dialogue.conversation_id: self.evaluate_dialogue(dialogue, **kwargs)
            for dialogue in dialogues
        }

    def evaluate_agents(
        self, dialogues: List[Dialogue], **kwargs: Any
    ) -> Dict[str, Dict[str, float]]:
        """Computes the metric for every agent over their dialogues.

        Args:
            dialogues: Dialogues.
            **kwargs: Additional arguments specific to the metric.

        Returns:
            Dictionary with result per agent.
        """
        dialogues_by_agent: Dict[str, List[Dialogue]] = defaultdict(list)
        for dialogue in dialogues:
            dialogues_by_agent[dialogue.agent_id].append(dialogue)
        return {
            agent_id: self.evaluate_dialogues(agent_dialogues, **kwargs)
            for agent_id, agent_dialogues in dialogues_by_agent.items()
        }
