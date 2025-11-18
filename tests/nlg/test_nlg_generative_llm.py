"""Tests for LLM-based natural language generation."""

import _io

from typing import List
from unittest.mock import mock_open
import pytest

from usersimcrs.llm_interfaces.ollama_interface import OllamaLLMInterface
from usersimcrs.nlg.llm.nlg_generative_llm import LLMGenerativeNLG
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.slot_value_annotation import SlotValueAnnotation
from dialoguekit.core.intent import Intent


@pytest.fixture
def llm_generative_nlg(
    monkeypatch, mock_ollama_interface: OllamaLLMInterface
) -> LLMGenerativeNLG:
    """LLMGenerativeNLG fixture."""
    prompt_file = "tests/data/nlg/test_generative_nlg_prompt.txt"

    monkeypatch.setattr("os.path.exists", lambda x: x == prompt_file)
    monkeypatch.setattr(
        "builtins.open",
        lambda file, mode="r": (
            mock_open(read_data="This is a mock prompt.").return_value
            if file == prompt_file
            else _io.open(file, mode)
        ),
    )

    return LLMGenerativeNLG(
        llm_interface=mock_ollama_interface,
        prompt_file=prompt_file,
    )


@pytest.mark.parametrize(
    "dialogue_acts, stringified_dialogue_acts",
    [
        (
            [
                DialogueAct(
                    intent=Intent("inform"),
                    annotations=[
                        SlotValueAnnotation("GENRE", "Comedy"),
                        SlotValueAnnotation("DIRECTOR", "Steven Spielberg"),
                    ],
                )
            ],
            "inform(GENRE=Comedy,DIRECTOR=Steven Spielberg)",
        ),
        (
            [
                DialogueAct(
                    intent=Intent("request"),
                    annotations=[SlotValueAnnotation("PLOT", None)],
                )
            ],
            "request(PLOT)",
        ),
        (
            [
                DialogueAct(
                    intent=Intent("inform"),
                    annotations=[SlotValueAnnotation("RATING", 4.5)],
                ),
                DialogueAct(
                    intent=Intent("request"),
                    annotations=[SlotValueAnnotation("PLOT", None)],
                ),
            ],
            "inform(RATING=4.5)|request(PLOT)",
        ),
        (
            [DialogueAct(intent=Intent("greet"))],
            "greet()",
        ),
    ],
)
def test_stringified_dialogue_acts(
    dialogue_acts: List[DialogueAct],
    stringified_dialogue_acts: str,
    llm_generative_nlg: LLMGenerativeNLG,
) -> None:
    assert (
        llm_generative_nlg._stringify_dialogue_acts(dialogue_acts)
        == stringified_dialogue_acts
    )


def test_generate_utterance_text(llm_generative_nlg: LLMGenerativeNLG) -> None:
    dialogue_acts = [
        DialogueAct(
            intent=Intent("inform"),
            annotations=[
                SlotValueAnnotation("GENRE", "Comedy"),
                SlotValueAnnotation("DIRECTOR", "Steven Spielberg"),
            ],
        ),
        DialogueAct(
            intent=Intent("request"),
            annotations=[
                SlotValueAnnotation("PLOT", None),
            ],
        ),
    ]

    llm_generative_nlg.llm_interface.get_llm_api_response.return_value = (
        "This is a generated utterance."
    )

    utterance = llm_generative_nlg.generate_utterance_text(
        dialogue_acts=dialogue_acts,
    )

    assert utterance.text == "This is a generated utterance."
    llm_generative_nlg.llm_interface.get_llm_api_response.assert_called_once_with(  # noqa: E501
        "This is a mock prompt."
    )


def test_generate_utterance_text_failed(
    llm_generative_nlg: LLMGenerativeNLG,
) -> None:
    llm_generative_nlg.llm_interface.get_llm_api_response.side_effect = (
        Exception("LLM API error")
    )
    dialogue_acts = [
        DialogueAct(
            intent=Intent("inform"),
            annotations=[
                SlotValueAnnotation("GENRE", "Comedy"),
            ],
        )
    ]
    with pytest.raises(
        RuntimeError, match="Failed to generate utterance: LLM API error"
    ):
        llm_generative_nlg.generate_utterance_text(
            dialogue_acts=dialogue_acts,
        )
