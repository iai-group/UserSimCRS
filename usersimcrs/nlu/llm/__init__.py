"""Module level init for LLM-based NLU components."""

"""Module level init for LLM-based NLU components.

Avoid importing heavy submodules at package import time to keep test
collection lightweight; import submodules explicitly when needed.
"""

__all__ = ["LLMDialogueActsExtractor"]
