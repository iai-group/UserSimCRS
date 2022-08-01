"""Context model."""


from typing import Any, Dict


class ContextModel:
    """Represents a context model."""

    _DEFAULT_CONTEXT_PROBABILITIES = dict()

    def __init__(
        self, context_probability_mapping: Dict[Any, Any] = None
    ) -> None:
        """Instantiates a context model with temporal and relational context.

        Args:
            context_probability_mapping: A dictionary with necessary
            probabilities to sample context. If it is not provided, we use
            default values.
        """
        pass

    def sample_context(self):
        """Samples a temporal and relational context.

        Args:

        Returns:
        """
        pass

    def _sample_temporal_context(self):
        """Samples a temporal context."""
        pass

    def _sample_relational_context(self):
        """Samples a relational context"""
