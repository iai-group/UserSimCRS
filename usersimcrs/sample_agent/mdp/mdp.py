from typing import List

import numpy as np
from dialoguekit.core.intent import Intent


class MDP:
    def __init__(
        self,
        states: List[Intent],
        actions: List[str],
        transition_model: np.array,
        policy: List[int] = None,
    ) -> None:
        """Instantiates MDP.

        Args:
            states: List of states.
            actions: List of actions.
            transition_model: Transition model.
            policy (optional): Policy.
        """
        self._states = states
        self._num_states = len(states)
        self._actions = actions
        self._num_actions = len(actions)
        assert transition_model.shape == (
            self._num_states,
            self._num_actions,
            self._num_states,
        )
        self._transition_model = transition_model
        self._policy = policy if policy else self.generate_random_policy()

    def generate_random_policy(self) -> np.array:
        """Generates a random policy."""
        return np.random.randint(self.num_actions, size=self.num_states)

    def next_intent(self, current_intent: Intent) -> Intent:
        """Predicts the next agent intent.

        Returns:
            Next agent intent based on probability distribution.
        """
        idx = self._states.index(current_intent)
        probabilities = self._transition_model[idx, self._policy[idx]]
        next_intent_idx = np.random.choice(self._num_states, p=probabilities)

        return self._states[next_intent_idx]
