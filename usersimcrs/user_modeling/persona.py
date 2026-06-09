from __future__ import annotations

"""Persona, which is a profile of the user to represent different backgrounds
(e.g., age, gender, education), personality types, and behavioral tendencies
(e.g., patience, conscientiousness, or curiosity)."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Persona:

    characteristics: Dict[str, Any]
    persona_description: Optional[str] = None

    @classmethod
    def from_config(cls, persona_config: Dict[str, Any]) -> Persona:
        """Creates a persona from a configuration dictionary.

        The configuration must provide characteristics under
        ``characteristics`` and may include an optional ``persona_description``.

        Args:
            persona_config: Persona configuration.

        Returns:
            Persona instance.
        """
        return cls(
            persona_config.get("characteristics", {}),
            persona_config.get("persona_description"),
        )
