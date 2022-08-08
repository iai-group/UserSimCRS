"""Persona."""

from dataclasses import dataclass


@dataclass
class Persona:
    """Persona dataclass. This dataclass is used to store user metadata, such as
    age, gender, education, and so on.
    TODO: Write why the metadata is significant for simulation."""

    age: int
    gender: str
    education_level: str
    cooperativeness: float
