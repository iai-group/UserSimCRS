"""Persona."""

from usersimcrs.user_modeling.context_model import ContextModel


class Persona:
    def __init__(self, cooperativeness: int) -> None:
        """Initializes a persona with cooperativeness etc."""
        self._cooperativeness = cooperativeness
        pass

    def _compute_patience(self, context: ContextModel) -> int:
        """Calculates persona patience based on cooperativeness and context.

        Args:
            context: An initialised context model.

        Returns:
            patience: An integer indicating number of retries befoore running
            out of patience.
        """
        pass
