"""Module level init for LLM interfaces."""

from usersimcrs.llm_interfaces.llm_interface import LLMInterface
from usersimcrs.llm_interfaces.openai_interface import OpenAILLMInterface
from usersimcrs.llm_interfaces.ollama_interface import OllamaLLMInterface

__all__ = [
    "LLMInterface",
    "OpenAILLMInterface",
    "OllamaLLMInterface",
]