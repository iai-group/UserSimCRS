"""Base class for utility metrics that require NLU annotation."""

from abc import ABC
from typing import Any, List, Optional, Tuple

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.intent import Intent
from dialoguekit.nlu.nlu import NLU

from usersimcrs.evaluation.base_metric import BaseMetric
from usersimcrs.evaluation.dialogue_annotation import (
    get_intent_lists,
    prepare_dialogue,
)


class UtilityBaseMetric(BaseMetric, ABC):
    """Shared base for metrics that optionally annotate dialogues via NLU.

    When NLU config paths are provided, dialogues are annotated automatically.
    When omitted, dialogues must be pre-annotated (e.g., via
    :func:`annotate_dialogues`).
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
            (
                _,
                rec,
                acc,
                rej,
                self._user_nlu,
                self._agent_nlu,
            ) = prepare_dialogue(
                dialogue,
                self._user_nlu_config_path,
                self._agent_nlu_config_path,
                self._user_nlu,
                self._agent_nlu,
                **kwargs,
            )
            return rec, acc, rej
        return get_intent_lists(**kwargs)
