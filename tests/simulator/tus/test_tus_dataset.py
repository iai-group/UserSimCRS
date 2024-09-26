"""Tests for preparing TUS dataset for training."""

from typing import List

import pytest

from dialoguekit.core.dialogue import Dialogue
from dialoguekit.utils.dialogue_reader import json_to_dialogues
from usersimcrs.simulator.neural.tus.tus_dataset import TUSDataset
from usersimcrs.simulator.neural.tus.tus_feature_handler import (
    TUSFeatureHandler,
)

TEST_DATA_PATH = "tests/data/tus_annotated_dialogues.json"


@pytest.fixture
def dialogues() -> List[Dialogue]:
    """Loads dialogues from the test data file."""
    return json_to_dialogues(TEST_DATA_PATH)


@pytest.fixture
def tus_dataset(feature_handler: TUSFeatureHandler) -> TUSDataset:
    """Returns a TUSDataset instance."""
    return TUSDataset(data_path=TEST_DATA_PATH, feature_handler=feature_handler)


def test_init_failure_file_path(feature_handler: TUSFeatureHandler) -> None:
    """Tests dataset initialization failure."""
    with pytest.raises(FileNotFoundError):
        TUSDataset(
            data_path="non_existent_file.json", feature_handler=feature_handler
        )


def test_init_failure_dialogue_format(
    feature_handler: TUSFeatureHandler,
) -> None:
    """Tests dataset initialization failure."""
    with pytest.raises(ValueError):
        TUSDataset(
            data_path="tests/data/annotated_dialogues.json",
            feature_handler=feature_handler,
        )


def test_len(tus_dataset: TUSDataset, dialogues: List[Dialogue]) -> None:
    """Tests the length of the dataset."""
    assert len(tus_dataset) == len(dialogues)


def test_process_dialogue(
    tus_dataset: TUSDataset, dialogues: List[Dialogue]
) -> None:
    """Tests dialogue processing."""
    dialogue = dialogues[0]

    input_features = tus_dataset.process_dialogue(dialogue)

    assert input_features.get("dialogue_id") == dialogue.conversation_id
    assert (
        len(input_features.get("input"))
        == len(input_features.get("mask"))
        == len(input_features.get("label"))
        == len(dialogue.utterances) // 2
    )
    assert all(
        len(input_features.get("input")[i])
        == len(input_features.get("mask")[i])
        == 80
        for i in range(len(input_features.get("input")))
    )
    assert all(
        len(input_features.get("label")[i]) == 40
        for i in range(len(input_features.get("label")))
    )
