"""Tests for LLM-based dialogue act extractor."""

import _io

from typing import List
from unittest.mock import mock_open
import pytest
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.intent import Intent
from dialoguekit.core.slot_value_annotation import SlotValueAnnotation
from usersimcrs.llm_interfaces.ollama_interface import OllamaLLMInterface
from usersimcrs.nlu.llm.llm_dialogue_act_extractor import (
    LLMDialogueActsExtractor,
)
from dialoguekit.core.utterance import Utterance
from dialoguekit.participant import DialogueParticipant


@pytest.fixture
def llm_dialogue_act_extractor(
    monkeypatch, mock_ollama_interface: OllamaLLMInterface
) -> LLMDialogueActsExtractor:
    """LLMDialogueActsExtractor fixture."""
    config_file = "tests/data/llm_dialogue_act_extractor_config.yaml"

    monkeypatch.setattr(
        "os.path.exists",
        lambda x: x == "tests/data/test_user_dact_extraction.txt"
        or x == config_file,
    )
    mock_prompt_file_content = "This is a mock extraction prompt."
    monkeypatch.setattr(
        "builtins.open",
        lambda file, mode="r": (
            mock_open(read_data=mock_prompt_file_content).return_value
            if file == "tests/data/test_user_dact_extraction.txt"
            else _io.open(file, mode)
        ),
    )

    return LLMDialogueActsExtractor(
        llm_interface=mock_ollama_interface,
        config_file=config_file,
    )


@pytest.mark.parametrize(
    "utterance, dialogue_acts, expected_filtered_acts",
    [
        (
            "I am looking for a recipe with rosemary.",
            [
                DialogueAct(
                    Intent("disclose"),
                    [
                        SlotValueAnnotation("ingredient", "rosemary"),
                    ],
                )
            ],
            [
                DialogueAct(
                    Intent("disclose"),
                    [
                        SlotValueAnnotation("ingredient", "rosemary"),
                    ],
                ),
            ],
        ),
        (
            "Can you help me find a recipe with rosemary?",
            [
                DialogueAct(
                    Intent("disclose"),
                    [
                        SlotValueAnnotation("ingredient", "thyme"),
                    ],
                )
            ],
            [
                DialogueAct(
                    Intent("disclose"),
                    [],
                ),
            ],
        ),
        (
            "I want to buy a new laptop.",
            [
                DialogueAct(
                    Intent("request"),
                    [
                        SlotValueAnnotation("item", "laptop"),
                    ],
                )
            ],
            [],
        ),
    ],
)
def test_filter_invalid_dialogue_acts(
    utterance: str,
    dialogue_acts: List[DialogueAct],
    expected_filtered_acts: List[DialogueAct],
    llm_dialogue_act_extractor: LLMDialogueActsExtractor,
) -> None:
    filtered_acts = llm_dialogue_act_extractor.filter_invalid_dialogue_acts(
        utterance,
        dialogue_acts,
    )
    assert all(
        filtered_act in expected_filtered_acts for filtered_act in filtered_acts
    )


@pytest.mark.parametrize(
    "utterance, mocked_llm_output, expected_dialogue_acts",
    [
        (
            Utterance(
                "I am looking for an Italian recipe.",
                participant=DialogueParticipant.USER,
            ),
            'disclose(cuisine="Italian")',
            [
                DialogueAct(
                    Intent("disclose"),
                    [
                        SlotValueAnnotation("cuisine", "Italian"),
                    ],
                )
            ],
        ),
        (
            Utterance(
                "Recommend me a Greek recipe with feta cheese.",
                participant=DialogueParticipant.USER,
            ),
            'disclose(cuisine="Greek",ingredient="feta cheese")',
            [
                DialogueAct(
                    Intent("disclose"),
                    [
                        SlotValueAnnotation("cuisine", "Greek"),
                        SlotValueAnnotation("ingredient", "feta cheese"),
                    ],
                )
            ],
        ),
    ],
)
def test_extract_dialogue_acts(
    utterance: Utterance,
    mocked_llm_output: str,
    expected_dialogue_acts: List[DialogueAct],
    llm_dialogue_act_extractor: LLMDialogueActsExtractor,
) -> None:
    """Tests dialogue act extraction."""
    llm_dialogue_act_extractor.llm_interface.get_llm_api_response.return_value = (  # noqa: E501
        mocked_llm_output
    )
    extracted_acts = llm_dialogue_act_extractor.extract_dialogue_acts(utterance)

    assert len(extracted_acts) == len(expected_dialogue_acts)
