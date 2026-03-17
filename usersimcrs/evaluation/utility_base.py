"""Base class for utility-centric dialogue evaluation metrics.

Provides shared NLU loading, and dialogue annotation.
"""

from abc import ABC
from typing import Optional

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.nlu.nlu import NLU

from usersimcrs.evaluation.base_metric import BaseMetric
from usersimcrs.evaluation.dialogue_annotation import (
    annotate_dialogue,
    load_nlu,
)

DEFAULT_REC_LABELS = ["REC-S", "REC-E"]
DEFAULT_ACC_LABELS = ["ACC"]
DEFAULT_REJ_LABELS = ["REJ"]


class UtilityBase(BaseMetric, ABC):
    def __init__(
        self,
        name: str,
        user_nlu_config_path: Optional[str] = None,
        agent_nlu_config_path: Optional[str] = None,
    ) -> None:
        """Initializes the utility metric.

        Args:
            name: Metric name.
            user_nlu_config_path: Path to user NLU configuration.
            agent_nlu_config_path: Path to agent NLU configuration.
        """
        super().__init__(name)
        self._user_nlu_config_path = user_nlu_config_path
        self._agent_nlu_config_path = agent_nlu_config_path
        self._user_nlu: Optional[NLU] = None
        self._agent_nlu: Optional[NLU] = None

    def _annotate_if_needed(self, dialogue: Dialogue) -> None:
        """Annotates the dialogue with NLU if config paths are set."""
        if self._user_nlu_config_path and self._agent_nlu_config_path:
            if self._user_nlu is None:
                self._user_nlu = load_nlu(
                    self._user_nlu_config_path,
                    "User NLU Configuration",
                )
            if self._agent_nlu is None:
                self._agent_nlu = load_nlu(
                    self._agent_nlu_config_path,
                    "Agent NLU Configuration",
                )
            annotate_dialogue(dialogue, self._user_nlu, self._agent_nlu)
