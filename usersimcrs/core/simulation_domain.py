"""Simulation domain knowledge.

This domain knowledge allows the definition of requestable and informable slots.
If not specified, all the slots are considered requestable and informable.
"""

from typing import List

from dialoguekit.core.domain import Domain


class SimulationDomain(Domain):
    def __init__(self, config_file: str) -> None:
        """Initializes the domain knowledge.

        Args:
            config_file: Path to the domain configuration file.
        """
        super().__init__(config_file)

    def get_requestable_slots(self) -> List[str]:
        """Returns the list of requestable slots."""
        if "requestable_slots" not in self._config:
            return self.get_slot_names()
        return self._config["requestable_slots"]

    def get_informable_slots(self) -> List[str]:
        """Returns the list of informable slots."""
        if "informable_slots" not in self._config:
            return self.get_slot_names()
        return self._config["informable_slots"]
