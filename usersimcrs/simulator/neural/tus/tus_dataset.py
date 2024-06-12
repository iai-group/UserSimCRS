"""Dataset class to encapsulate processing of data for training TUS.

The data is expected to follow DialogueKit's format in addition to an
information need per dialogue.
"""

import os
from typing import Any, Dict, List

import torch
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.participant import DialogueParticipant
from dialoguekit.utils.dialogue_reader import json_to_dialogues

from usersimcrs.core.information_need import InformationNeed
from usersimcrs.dialogue_management.dialogue_state_tracker import (
    DialogueStateTracker,
)
from usersimcrs.simulator.neural.tus.tus_feature_handler import (
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

        Raises:
            ValueError: If information need is not found in the dialogue
              metadata.

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
        information_need = dialogue.metadata.get("information_need", None)
        if information_need is None:
            raise ValueError("Information need not found in dialogue metadata.")
        information_need: InformationNeed = InformationNeed.from_dict(
            information_need
        )

        last_user_actions = {}
        utterances = dialogue.utterances
        for i, utterance in enumerate(utterances):
            if utterance.participant == DialogueParticipant.AGENT.name:
                dst.update_state(
                    utterance.dialogue_acts, DialogueParticipant.AGENT
                )
                continue

            agent_dialogue_acts = (
                utterances[i - 1].dialogue_acts if i > 0 else []
            )
            feature_vector, mask = self.feature_handler.build_input_vector(
                agent_dialogue_acts,
                previous_state,
                dst.get_current_state(),
                information_need,
                last_user_actions,
            )
            label = self.feature_handler.get_label_vector(
                utterance, dst.get_current_state(), information_need
            )
            input_representations["input"].append(feature_vector)
            input_representations["mask"].append(mask)
            input_representations["label"].append(label)

            dst.update_state(utterance.dialogue_acts, DialogueParticipant.USER)
            previous_state = dst.get_current_state()

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
            self.feature_handler.reset_user_feature_history()
            processed_dialogue = self.process_dialogue(dialogue)
            for key, value in processed_dialogue.items():
                processed_data[key].extend(value)

        processed_data["input"] = torch.tensor(
            processed_data["input"], dtype=torch.float32
        )
        processed_data["mask"] = torch.tensor(
            processed_data["mask"], dtype=torch.float32
        )
        processed_data["label"] = torch.tensor(
            processed_data["label"], dtype=torch.long
        )
        processed_data["mask"] = torch.tensor(
            processed_data["mask"], dtype=torch.bool
        )

        return processed_data
