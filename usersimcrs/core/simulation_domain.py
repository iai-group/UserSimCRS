"""Simulation domain knowledge."""

from typing import List

from dialoguekit.core.domain import Domain


class SimulationDomain(Domain):
    def __init__(self, config_file: str) -> None:
        """Initializes the domain knowledge.

        Args:
            config_file: Path to the domain configuration file.

        Raises:
            KeyError: If the domain configuration file does not contain the
              field 'requestable_slots'.
        """
        super().__init__(config_file)
        if "requestable_slots" not in self._config:
            raise KeyError(
                "The domain configuration file must contain the field "
                "'requestable_slots'."
            )

    def get_requestable_slots(self) -> List[str]:
        """Returns the list of requestable slots."""
        return self._config["requestable_slots"]
