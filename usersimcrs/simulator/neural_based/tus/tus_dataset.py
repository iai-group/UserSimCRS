"""Dataset class to encapsulate processing of data for training TUS.

The data is expected to follow DialogueKit's format in addition to an
information need per dialogue."""

import os
from typing import Any, Dict, List

import torch
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.participant import DialogueParticipant
from dialoguekit.utils.dialogue_reader import json_to_dialogues

from usersimcrs.dialogue_management.dialogue_state_tracker import (
    DialogueStateTracker,
)
from usersimcrs.simulator.neural_based.tus.tus_feature_handler import (
    TUSFeatureHandler,
)


class TUSDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        feature_handler: TUSFeatureHandler,
        agent_ids: List[str] = None,
        user_ids: List[str] = None,
    ) -> None:
        """Initializes the dataset.

        Args:
            data_path: Path to the data file.
            feature_handler: Feature handler.
            agent_ids: List of agents' id to filter loaded dialogues. Defaults
              to None.
            user_ids: List of users' id to filter loaded dialogues. Defaults to
              None.

        Raises:
            FileNotFoundError: If the data file is not found.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File '{data_path}' not found.")

        self.raw_data = json_to_dialogues(
            data_path,
            agent_ids=agent_ids,
            user_ids=user_ids,
        )

        self.feature_handler = feature_handler
        self.input_vectors = self.process_dialogues()

    def __len__(self) -> int:
        """Returns the number of dialogues in the dataset."""
        return len(self.raw_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retrieves input representation for a given utterance.

        Args:
            idx: Index of the utterance.

        Returns:
            Input representation.
        """
        return {
            "input": self.input_vectors["input"][idx],
            "mask": self.input_vectors["mask"][idx],
            "label": self.input_vectors["label"][idx],
            "dialogue_id": self.input_vectors["dialogue_id"][idx],
        }

    def process_dialogue(self, dialogue: Dialogue) -> Dict[str, Any]:
        """Processes a dialogue to create input representations.

        Args:
            dialogue: Dialogue to process.

        Returns:
            Input representation for each user utterance in the dialogue.
        """
        input_representations = {
            "input": [],
            "mask": [],
            "label": [],
            "dialogue_id": dialogue.conversation_id,
        }

        self.feature_handler.reset()
        dst = DialogueStateTracker()
        previous_state = dst.get_current_state()
        information_need = dialogue.information_need
        last_user_actions = {}
        for utterance in dialogue.utterances:
            if utterance.participant == DialogueParticipant.AGENT:
                dst.update_state(
                    utterance.dialogue_acts, DialogueParticipant.AGENT
                )
                continue

            feature_vector = self.feature_handler.get_feature_vector(
                utterance,
                previous_state,
                dst.get_current_state(),
                information_need,
                last_user_actions,
            )
            label = self.feature_handler.get_label_vector(
                utterance, dst.get_current_state(), information_need
            )
            input_representations["input"].append(feature_vector)
            # TODO
            # input_representations["mask"].append()
            input_representations["label"].append(label)

            dst.update_state(utterance.dialogue_acts, DialogueParticipant.USER)
            previous_state = dst.get_current_state()

        for key in input_representations:
            input_representations[key] = torch.cat(input_representations[key])

        return input_representations

    def process_dialogues(self) -> Dict[str, Any]:
        """Processes dialogues to create input representations.

        Returns:
            Processed dialogues.
        """
        processed_data = {
            "input": [],
            "mask": [],
            "label": [],
            "dialogue_id": [],
        }
        for dialogue in self.raw_data:
            processed_dialogue = self.process_dialogue(dialogue)
            for key, value in processed_dialogue.items():
                processed_data[key].append(value)

        return processed_data
