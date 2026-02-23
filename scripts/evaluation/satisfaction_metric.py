"""Satisfaction metric class implementation.

Wraps DialogueKit's satisfaction classifier into a `BaseMetric` class.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dialoguekit.core.dialogue import Dialogue  # type: ignore
    from dialoguekit.nlu.models.satisfaction_classifier import (
        SatisfactionClassifierSVM,
    )  # type: ignore
else:
    try:
        from dialoguekit.core.dialogue import Dialogue
        from dialoguekit.nlu.models.satisfaction_classifier import (
            SatisfactionClassifierSVM,
        )
    except Exception:
        Dialogue = Any
        SatisfactionClassifierSVM = Any

from scripts.evaluation.base_metric import BaseMetric


class SatisfactionMetric(BaseMetric):
    """Wraps the `SatisfactionClassifierSVM` to compute satisfaction scores.

    Output format matches previous CLI script: { agent_id: { dialogue_index:
    score, ... }, ... }
    """

    def __init__(self, classifier: Optional[SatisfactionClassifierSVM] = None):
        super().__init__()
        self.classifier = classifier or SatisfactionClassifierSVM()

    @property
    def name(self) -> str:
        return "satisfaction"

    def compute(self, dialogues: List[Dialogue]) -> Dict[str, Dict[int, float]]:
        """Compute satisfaction scores for dialogues.

        Matches the previous CLI output format: agent_id -> dialogue_index ->
        score
        """
        scores: Dict[str, Dict[int, float]] = defaultdict(dict)
        for i, dialogue in enumerate(dialogues):
            scores[dialogue.agent_id][
                i
            ] = self.classifier.classify_last_n_dialogue(dialogue, last_n=None)
        return scores


class SatisfactionAverageMetric(SatisfactionMetric):
    """Aggregates satisfaction scores and returns average per agent."""

    @property
    def name(self) -> str:
        return "satisfaction.average"

    def compute(self, dialogues: List[Dialogue]) -> Dict[str, float]:
        raw = super().compute(dialogues)
        averages: Dict[str, float] = {}
        for agent_id, agent_scores in raw.items():
            if len(agent_scores) == 0:
                averages[agent_id] = 0.0
            else:
                averages[agent_id] = sum(agent_scores.values()) / len(
                    agent_scores
                )
        return averages
