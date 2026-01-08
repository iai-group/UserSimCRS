"""Module level init for prompts."""

from usersimcrs.simulator.llm.prompt.stop_prompt import StopPrompt
from usersimcrs.simulator.llm.prompt.utterance_generation_prompt import UtteranceGenerationPrompt

__all__ = [
    "StopPrompt",
    "UtteranceGenerationPrompt",
]