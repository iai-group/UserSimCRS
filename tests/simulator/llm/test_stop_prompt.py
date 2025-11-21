"""Tests stop prompt."""

import pytest

from usersimcrs.core.information_need import InformationNeed
from usersimcrs.simulator.llm.prompt.stop_prompt import (
    DEFAULT_STOP_DEFINITION,
    StopPrompt,
)


@pytest.fixture
def prompt(information_need: InformationNeed) -> StopPrompt:
    """Returns a Prompt object."""
    return StopPrompt(information_need, "item")


def test_build_new_prompt(prompt: StopPrompt) -> None:
    """Tests the build_new_prompt method."""
    assert prompt.build_new_prompt() == (
        DEFAULT_STOP_DEFINITION + "\nHISTORY:\n"
    )


def test_prompt_text(prompt: StopPrompt) -> None:
    """Tests the prompt_text property."""
    initial_prompt = f"{DEFAULT_STOP_DEFINITION}\nHISTORY:\n"

    assert prompt.prompt_text == initial_prompt + "\n\nCONTINUE: "
