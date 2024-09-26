"""Tests prompt."""

import pytest
from dialoguekit.core import Utterance
from dialoguekit.participant import DialogueParticipant

from usersimcrs.core.information_need import InformationNeed
from usersimcrs.simulator.llm.prompt import DEFAULT_TASK_DEFINITION, Prompt
from usersimcrs.user_modeling.persona import Persona


@pytest.fixture
def persona() -> Persona:
    """Returns a Persona object."""
    return Persona(characteristics={"curiosity": "high", "education": "MSc"})


@pytest.fixture
def prompt(information_need: InformationNeed, persona: Persona) -> Prompt:
    """Returns a Prompt object."""
    return Prompt(information_need, "item", persona=persona)


def test_build_new_prompt(prompt: Prompt) -> None:
    """Tests the build_new_prompt method."""
    stringified_persona = (
        " Adapt your responses considering your PERSONA.\nPERSONA: "
        "curiosity=high, education=MSc\n"
    )
    stringified_requirements = (
        "REQUIREMENTS: You are looking for a item with the following "
        "characteristics: GENRE=Comedy, DIRECTOR=Steven Spielberg. Once you "
        "find a suitable item, make sure to get the following information: "
        "PLOT, RATING.\n"
    )

    assert prompt.build_new_prompt() == (
        DEFAULT_TASK_DEFINITION + stringified_persona + stringified_requirements
    )


def test_update_prompt_context(prompt: Prompt) -> None:
    """Tests the update_prompt_context method."""
    user_utterance = Utterance(
        "I am looking for a comedy movie.", DialogueParticipant.USER
    )
    agent_utterance = Utterance(
        "I suggest 'The terminal'.", DialogueParticipant.AGENT
    )

    prompt.update_prompt_context(user_utterance, DialogueParticipant.USER)
    assert prompt._prompt_context == "USER: I am looking for a comedy movie.\n"
    prompt.update_prompt_context(agent_utterance, DialogueParticipant.AGENT)
    assert prompt._prompt_context == (
        "USER: I am looking for a comedy movie.\nASSISTANT: I suggest 'The "
        "terminal'.\nUSER: "
    )
