class InteractionModel:
    """Represents an interaction model."""

    def __init__(self) -> None:
        """Initializes the interaction model."""
        self._agenda = []
        self._current_intent = None
        pass

    def initialize_agenda(self):
        """Initializes the action agenda."""
        pass

    @property
    def agenda(self):
        return self._agenda

    @property
    def current_intent(self):
        return self._current_intent

    def update_agenda(self, intent: str):
        """Updates the agenda and determins the next user intent based on agent intent.

        If agent replies with right intent in response to last user intent (checking mapping):
            pops up the next user intent from the agenda;
            update the current intent;
        else:
            push a new intent (select a replacement intent);
            update the current intent;
        """
