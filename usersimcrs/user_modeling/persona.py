"""Persona, which is a profile of the user to represent different backgrounds
(e.g., age, gender, education), personality types, and behavioral tendencies
(e.g., patience, conscientiousness, or curiosity)."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Persona:
    """Represents personal user characteristics as key-value pairs."""

    characteristics: Dict[str, Any]
