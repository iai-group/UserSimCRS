"""Persona."""

from dataclasses import dataclass


@dataclass
class Persona:
    age: int
    gender: str
    education_level: str
    cooperativeness: float
