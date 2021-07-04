"""Interaction model."""

# TODO: change intent types to Intent once that's been added to dialoguekit.core

import os
import yaml
import random
from typing import List, Dict, Any, Tuple


class InteractionModel:
    """Represents an interaction model."""

    START_INTENT = "DISCLOSE.NON-DISCLOSE"
    STOP_INTENT = "STOP"

    def __init__(self, config_file, annotated_conversations: List[List]) -> None:
        """Initializes the interaction model."""
        # Load interaction model.
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with open(config_file) as yaml_file:
            self._config = yaml.load(yaml_file, Loader=yaml.FullLoader)

        (
            self._user_intent_distribution,
            self._intent_distribution,
        ) = self.intent_distribution(annotated_conversations)
        # Initialize agenda.
        self._agenda = self.initialize_agenda()
        # Keep track of the current user intent.
        self._current_intent = self._agenda.pop()

    def intent_distribution(
        self, annotated_conversations: List[Dict]
    ) -> Tuple[Dict, Dict]:
        """Distill user intent distributions based on conversations.

        Arg:
            Annotated_conversations: list of annotated conversations.

        Returns:
            Intent distributions: {user of agent intent: {next_user_intent: occurrence}}
        """
        user_intent_dist, intent_dist = dict(), dict()
        for annotated_conversation in annotated_conversations:
            user_agenda = [
                u["intent"]
                for u in annotated_conversation["conversation"]
                if u["participant"] == "USER"
            ]
            # Makes sure all the user agenda end with "COMPLETE".
            if user_agenda[-1] != self.STOP_INTENT:
                user_agenda.append(self.STOP_INTENT)
            for i, user_intent in enumerate(user_agenda):
                if user_intent not in user_intent_dist:
                    user_intent_dist[user_intent] = dict()
                next_user_intent = (
                    user_agenda[i + 1] if i < len(user_agenda) - 1 else self.STOP_INTENT
                )
                if next_user_intent not in user_intent_dist[user_intent]:
                    user_intent_dist[user_intent][next_user_intent] = 0
                user_intent_dist[user_intent][next_user_intent] += 1

            for j, utterance_record in enumerate(
                annotated_conversation["conversation"]
            ):
                # Only consider agent intent as keys
                if utterance_record["participant"] != "AGENT":
                    continue
                intent = utterance_record["intent"]
                if intent not in intent_dist:
                    intent_dist[intent] = dict()
                # TODO: consider the case when the next intent is not user intent.
                next_user_intent = (
                    annotated_conversation["conversation"][j + 1]["intent"]
                    if j < len(annotated_conversation["conversation"]) - 1
                    else self.START_INTENT
                )
                if next_user_intent not in intent_dist[intent]:
                    intent_dist[intent][next_user_intent] = 0
                intent_dist[intent][next_user_intent] += 1

        return user_intent_dist, intent_dist

    def initialize_agenda(self) -> List:
        """Initializes the action agenda.

        Step1: Load all the dialogues with intents and generate a map:
                intent_map = {
                    current_intent:
                        {next_intent1: n_1,
                         next_intent_2: n_2}
                }
        Note: CIR6 only based on user intents while qrfa uses both user and agent intents

        Step2: populate the agenda.
            starting_intent = "None_disclose"
            self.agenda.append(starting_intent)
            next_intent = self.next_intent(starting_intent)
            agenda.append(next_intent)
            while next_intent != "stop":
                self..append(next_intent)
                next_intent = self.next_intent(next_intent)
            agenda.append(next_intent)

        Step3: filter the agenda, e.g. too short or too long agenda will triger this function rerun
        """
        current_intent_str = self.START_INTENT
        agenda = list()
        agenda.append(current_intent_str)
        next_intent_str = self.next_intent(
            current_intent_str, self._user_intent_distribution
        )
        while next_intent_str != self.STOP_INTENT:
            current_intent_str = next_intent_str
            agenda.append(current_intent_str)
            next_intent_str = self.next_intent(
                current_intent_str, self._user_intent_distribution
            )
        agenda.reverse()
        self._agenda = agenda
        return agenda

    @property
    def agenda(self):
        return self._agenda

    @property
    def current_intent(self) -> str:
        return self._current_intent

    def is_agent_intent_elicit(self, agent_intent: str) -> bool:
        """Checks if the given agent intent is elicitation.

        Args:
            agent_intent: Agent's intent.

        Returns:
            True if it is an elicitation intent.
        """
        return agent_intent in self._config["agent_elicit_intents"]

    def is_agent_intent_set_retrieval(self, agent_intent: str) -> bool:
        """Checks if the given agent intent is set retrieval.

        Args:
            agent_intent: Agent's intent.

        Returns:
            True if it is a set retrieval intent.
        """
        return agent_intent in self._config["agent_set_retrieval"]

    def next_intent(self, intent_str: str, intent_dist: dict) -> str:
        """Predicts the next user intent.

        Given current_intent, we determine the next intent (either next_intent1 or next_intent2) by probabilities.

        Args:
            Intent_str: str of current user intent.

        Returns:
            Next user intent string based on probability distribution.
        """
        # Get the distribution of next intent for the current user intent.
        intent_map = intent_dist.get(intent_str)
        assert isinstance(intent_map, dict)

        # Randomly generates a probability from 0~1.
        p_random = random.uniform(0, 1)

        # Get the sum of the next intent occurences.
        next_intent_occurences_sum = sum(intent_map.values())

        # Get normalized next intent distribution occurences and next intent list.
        d, next_intents = [], []
        for next_intent, next_intent_occurrence in intent_map.items():
            d.append(next_intent_occurrence / next_intent_occurences_sum)
            next_intents.append(next_intent)
        return self.p_sample(p_random, d, next_intents)

    @staticmethod
    def p_sample(p: float, d: List, items: List) -> Any:
        """Determines the next item based on a randomly generated probability.

        Args:
            p: a randomly generated uniform probability.
            d: list of probablitities of items.
            items: items to be sampled from.

        Return:
            The sampled item.
        """
        p_start = 0
        for i, p_item in enumerate(d):
            p_start += p_item
            if p < p_start:
                return items[i]
        return items[-1]

    def update_agenda(self, agent_intent: str) -> str:
        """Updates the agenda and determines the next user intent based on agent
        intent.

        If agent replies with an expected intent in response to the last user
        intent (based on the expected_responses mapping in the config file),
        then
            pops up the next user intent from the agenda;
            update the current intent;
        Otherwise:
            pushes a new intent (select a replacement intent);
            updates the current intent.

        Args:
            Agent_intent: Agent's intent

        Returns:
            The current intent (the next intent).
        """
        expected_agent_intents = self._config.get("expected_responses").get(
            self._current_intent
        )
        # If agent replies in an expected intent, then pop the next intent from agenda.
        if agent_intent in expected_agent_intents:
            self._current_intent = self._agenda.pop()
        else:  # Find a replacement based on last agent intent
            self._current_intent = self.next_intent(
                agent_intent, self._intent_distribution
            )
        print(self._agenda.pop())
        return self._current_intent
