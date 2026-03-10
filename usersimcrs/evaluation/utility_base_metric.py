"""Base class for dialogue annotation support."""

from abc import ABC
from typing import Any, List, Optional, Tuple

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.intent import Intent
from dialoguekit.nlu.nlu import NLU

from usersimcrs.evaluation.base_metric import BaseMetric
from usersimcrs.evaluation.dialogue_annotation import (
    annotate_dialogue,
    get_intent_lists,
    load_nlu,
)


class UtilityBaseMetric(BaseMetric, ABC):
    """Shared base for metrics that optionally annotate dialogues via NLU.

    When NLU config paths are provided, dialogues are annotated automatically.
    When omitted, dialogues must be pre-annotated.
    """

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

    def _resolve_intents(
        self, dialogue: Dialogue, **kwargs: Any
    ) -> Tuple[List[Intent], List[Intent], List[Intent]]:
        """Annotates the dialogue (if NLU paths are set) and returns intents.

        Args:
            dialogue: Dialogue to prepare.
            **kwargs: Optional intent label overrides forwarded to
                :func:`get_intent_lists`.

        Returns:
            Tuple of (recommendation_intents, acceptance_intents,
            rejection_intents).
        """
        if self._user_nlu_config_path and self._agent_nlu_config_path:
            self._user_nlu = load_nlu(
                self._user_nlu_config_path,
                "User NLU Configuration",
                self._user_nlu,
            )
            self._agent_nlu = load_nlu(
                self._agent_nlu_config_path,
                "Agent NLU Configuration",
                self._agent_nlu,
            )
            annotate_dialogue(dialogue, self._user_nlu, self._agent_nlu)
        return get_intent_lists(**kwargs)
