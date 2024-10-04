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
    stringified_requirements = (
        "\nREQUIREMENTS: You are looking for a item with the following "
        "characteristics: genre=Comedy, director=Steven Spielberg and want to "
        "know the following information about it: plot, rating.\nHISTORY:\n"
    )

    assert prompt.build_new_prompt() == (
        DEFAULT_STOP_DEFINITION + stringified_requirements
    )


def test_prompt_text(prompt: StopPrompt) -> None:
    """Tests the prompt_text property."""
    initial_prompt = (
        f"{DEFAULT_STOP_DEFINITION}\nREQUIREMENTS: You are looking "
        "for a item with the following characteristics: genre=Comedy, "
        "director=Steven Spielberg and want to know the following information "
        "about it: plot, rating.\nHISTORY:\n"
    )

    assert prompt.prompt_text == initial_prompt + "\n\nCONTINUE: "
