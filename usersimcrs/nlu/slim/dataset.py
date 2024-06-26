"""Dataset to train and evaluate the SLIM model."""

import json
import logging
import os
import re
import string
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Generator, List, Tuple

import torch
from seqeval.metrics.sequence_labeling import get_entities
from torch.utils.data import Dataset
from transformers import BertTokenizer

_TOKENIZER_NAME = "bert-base-uncased"
SPLIT_CHARS = r"([{}])".format(
    re.escape(string.punctuation + string.whitespace)
)


class Participant(Enum):
    USER = "USER"
    AGENT = "AGENT"
    ALL = "ALL"


def load_and_parse_data(
    path: str, participant: Participant, max_slots: int
) -> Generator:
    """Loads and parses dialogues to extract text, intents, and slots.

    The data is expected to match DialogueKit's format.

    Args:
        path: Path to the dataset.
        participant: Participant to filter utterances.
        max_slots: Maximum number of slots in an example.

    Raises:
        FileNotFoundError: If the file is not found.

    Yields:
        A tuple with the text, BIO sequences for intents and slots.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r") as file:
        raw_data = json.load(file)

    # Filter utterances based on participant
    utterances = []
    for dialogue in raw_data:
        for utterance in dialogue["conversation"]:
            if participant == Participant.ALL:
                utterances.append(utterance)
            elif utterance["participant"] == participant.value:
                utterances.append(utterance)
    logging.info(f"Loaded {len(utterances)} utterances.")

    # Parse utterances to extract text, BIO sequences for intents and slots
    for utterance in utterances:
        text, words, intent_bio, slot_bio = parse_utterance(
            utterance, max_slots
        )
        yield text, words, intent_bio, slot_bio


def parse_utterance(
    utterance: Dict[str, Any], max_slots: int
) -> Tuple[str, List[str], List[str], List[str]]:
    """Parses an utterance to extract text, BIO sequences for intents and slots.

    Args:
        utterance: Utterance to parse.
        max_slots: Maximum number of slots in an example.

    Returns:
        A tuple with the text, words, and BIO sequences for intents and slots.
    """
    text = utterance["utterance"].strip()
    words = re.split(SPLIT_CHARS, text)
    num_words = len(words)

    words += ["[HIDDEN]"] * max_slots
    intent_bio = ["O"] * len(words)
    slot_bio = ["O"] * len(words)
    for dialogue_act in utterance.get("dialogue_acts", []):
        intent = dialogue_act.get("intent", "UNK")
        for slot, value in dialogue_act.get("slot_values", []):
            if slot is None:
                continue

            if value is None:
                # The slot is hidden when it does not have a value, e.g.,
                # when it is elicited.
                i = intent_bio.index("O", num_words)
                intent_bio[num_words] = intent
                slot_bio[num_words] = f"B-{slot}"
                continue

            value_words = re.split(SPLIT_CHARS, value)
            partial_slot_bio = [f"B-{slot}"] + [f"I-{slot}"] * (
                len(value_words) - 1
            )
            partial_intent_bio = [intent] * (len(value_words))
            # Find the value in the utterance. Note that the current
            # implementation may overwrite a slot value if it appears
            # multiple times in the utterance.
            # TODO: Find better strategy for slot value extraction.
            for i, word in enumerate(words):
                if word == value_words[0]:
                    # fmt: off
                    if (
                        words[i : i + len(value_words)]  # noqa: E203
                        == value_words
                    ):
                        slot_bio[i : i + len(value_words)] = (  # noqa: E203
                            partial_slot_bio
                        )
                        intent_bio[i : i + len(value_words)] = (  # noqa: E203
                            partial_intent_bio
                        )
                    # fmt: on

    return text, words, intent_bio, slot_bio


@dataclass
class SLIMExample:
    """Represents an example for the SLIM model."""

    text: str
    words: List[str]
    intent_labels: List[str] = None
    slot_labels: List[str] = None  # BIO format
    tag_intent_mask: List[List[int]] = None
    tag_intent_labels: List[str] = None


class SLIMDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        intent_labels_map: Dict[str, int],
        slot_labels_map: Dict[str, int],
        participant: Participant,
        max_length: int = 128,
        max_slot: int = 6,
    ) -> None:
        """Initializes to train/evaluate the SLIM model.

        Args:
            data_path: Path to the dataset.
            intent_labels_map: Mapping of intent labels to integers.
            slot_labels_map: Mapping of slot labels to integers.
            participant: Participant to filter utterances.
            max_length: Maximum length of input sequence. Defaults to 128.
            max_slot: Maximum number of slots in an example. Defaults to 6.
        """
        self.data = load_and_parse_data(data_path, participant, max_slot)
        self.max_length = max_length
        self.max_slot = max_slot

        self.intent_labels_map = intent_labels_map
        self.slot_labels_map = slot_labels_map

        self.tokenizer = BertTokenizer.from_pretrained(_TOKENIZER_NAME)

        self.examples: List[SLIMExample] = []
        self._create_examples()

    def _create_examples(self) -> None:
        """Creates examples from the dataset."""
        for text, words, intent_bio, slot_bio in self.data:
            assert len(words) == len(slot_bio) == len(intent_bio)

            # Intent labels
            intent_labels = list(set(intent_bio) - {"O"})

            # Slot-intent mapping
            entities = get_entities(slot_bio)
            entities = entities[: self.max_slot]

            tag_intent_mask = [
                [0 for _ in slot_bio] for _ in range(self.max_slot)
            ]
            tag_intent_labels = ["PAD"] * self.max_slot
            try:
                for i, (_, start, end) in enumerate(entities):
                    tag_intent_mask[i][start : end + 1] = [  # noqa: E203
                        1 / (end - start + 1)
                    ] * (end - start + 1)
                    tag_intent_labels[i] = intent_bio[start]
                    assert tag_intent_labels[i] not in ["UNK", "O"]
            except Exception as e:
                logging.error(f"Error: {e}")
                logging.info(
                    "Skipping example due to alignment error when "
                    f"mapping slots to intents. Example: {text}"
                )
                continue

            example = SLIMExample(
                text=text,
                words=words,
                intent_labels=intent_labels,
                slot_labels=slot_bio,
                tag_intent_mask=tag_intent_mask,
                tag_intent_labels=tag_intent_labels,
            )
            self.examples.append(example)

    def _convert_example_to_features(
        self, example: SLIMExample
    ) -> Dict[str, torch.Tensor]:
        """Converts an example to features.

        The features include input_ids, attention_mask, token_type_ids,
        intent_label_ids, slot_label_ids, tag_intent_mask, and
        tag_intent_label_ids.

        Args:
            example: Example to convert.

        Returns:
            Feature representation of the example.
        """
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        unk_token = self.tokenizer.unk_token
        pad_token_id = self.tokenizer.pad_token_id
        pad_token_label_id = 0

        tokens = []
        slot_labels_ids = []
        tag_intent_mask_ids = []
        tag_intent_mask = list(zip(*example.tag_intent_mask))

        # Tokenization per word
        for word, slot_label, tag_intent_pos_mask in zip(
            example.words,
            example.slot_labels,
            tag_intent_mask,
        ):
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]
            tokens.extend(word_tokens)

            slot_labels_ids.extend(
                [self.slot_labels_map[slot_label]]
                + [pad_token_label_id] * (len(word_tokens) - 1)
            )
            tag_intent_mask_ids.extend(
                [tag_intent_pos_mask]
                + [tuple([0 for _ in range(self.max_slot)])]
                * (len(word_tokens) - 1)
            )

        # Truncate tokens and labels to max_length - 2 to account for [CLS]
        # and [SEP]
        tokens = tokens[: self.max_length - 2]
        slot_labels_ids = slot_labels_ids[: self.max_length - 2]
        tag_intent_mask_ids = tag_intent_mask_ids[: self.max_length - 2]

        tokens = [cls_token] + tokens + [sep_token]
        token_type_ids = [0] * len(tokens)
        slot_labels_ids = (
            [pad_token_label_id] + slot_labels_ids + [pad_token_label_id]
        )
        tag_intent_mask_ids = (
            [tuple([0 for _ in range(self.max_slot)])]
            + tag_intent_mask_ids
            + [tuple([0 for _ in range(self.max_slot)])]
        )

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        # Padding
        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + [pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        token_type_ids = token_type_ids + [0] * padding_length
        slot_labels_ids = (
            slot_labels_ids + [pad_token_label_id] * padding_length
        )
        tag_intent_mask_ids = (
            tag_intent_mask_ids
            + [tuple([0 for _ in range(self.max_slot)])] * padding_length
        )
        tag_intent_mask_ids = list(zip(*tag_intent_mask_ids))
        tag_intent_mask_ids = [
            list(tag_intent_mask_id)
            for tag_intent_mask_id in tag_intent_mask_ids
        ]

        # Intent labels
        intent_label_ids = [0] * len(self.intent_labels_map)
        for intent_label in example.intent_labels:
            intent_label_ids[self.intent_labels_map[intent_label]] = 1

        # Slot-intent mapping
        tag_intent_label_ids = [
            self.intent_labels_map[label] for label in example.tag_intent_labels
        ]

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "token_type_ids": torch.tensor(token_type_ids),
            "intent_label_ids": torch.tensor(
                intent_label_ids, dtype=torch.float32
            ),
            "slot_label_ids": torch.tensor(slot_labels_ids),
            "tag_intent_mask": torch.tensor(
                tag_intent_mask_ids, dtype=torch.float32
            ),
            "tag_intent_label_ids": torch.tensor(tag_intent_label_ids),
        }

    def __len__(self) -> int:
        """Returns the number of examples."""
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Returns the example at the given index.

        Args:
            index: Index of the example.

        Returns:
            A dictionary with input_ids, attention_mask, labels for intents,
              slots, and slot-intent mapping.
        """
        example = self.examples[index]
        return self._convert_example_to_features(example)
