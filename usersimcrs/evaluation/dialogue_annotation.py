"""Dialogue annotation and recommendation round utilities.

Provides functions for annotating dialogues with dialogue acts using NLU
modules, parsing intent labels, and extracting recommendation rounds from
annotated dialogues.
"""

from typing import Any, List, Optional, Tuple

from confuse import Configuration

from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue import Dialogue
from dialoguekit.core.intent import Intent
from dialoguekit.nlu.nlu import NLU
from dialoguekit.participant.participant import DialogueParticipant

from usersimcrs.utils.simulation_utils import get_NLU


def annotate_dialogue(
    dialogue: Dialogue, user_nlu: NLU, agent_nlu: NLU
) -> Dialogue:
    """Annotates utterances with dialogue acts.

    Each utterance that is not already an AnnotatedUtterance is converted to
    one.  Utterances that already carry dialogue acts are left untouched.

    Args:
        dialogue: Dialogue to be annotated.
        user_nlu: NLU module for user utterances.
        agent_nlu: NLU module for agent utterances.

    Raises:
        ValueError: If an utterance has an unknown participant.

    Returns:
        The same dialogue object with annotated utterances.
    """
    for i, utterance in enumerate(dialogue.utterances):
        if not isinstance(utterance, AnnotatedUtterance):
            dialogue.utterances[i] = AnnotatedUtterance.from_utterance(
                utterance
            )

        if len(utterance.dialogue_acts) > 0:
            continue

        if utterance.participant == DialogueParticipant.USER:
            dialogue.utterances[
                i
            ].dialogue_acts = user_nlu.extract_dialogue_acts(utterance)
        elif utterance.participant == DialogueParticipant.AGENT:
            dialogue.utterances[
                i
            ].dialogue_acts = agent_nlu.extract_dialogue_acts(utterance)
        else:
            raise ValueError(f"Unknown participant: {utterance.participant}")
    return dialogue


def load_nlus(
    user_nlu_config_path: str,
    agent_nlu_config_path: str,
    cached_user_nlu: Optional[NLU] = None,
    cached_agent_nlu: Optional[NLU] = None,
) -> Tuple[NLU, NLU]:
    """Loads user and agent NLU modules.

    Returns cached instances when provided, otherwise creates new ones
    from the given configuration files.

    Args:
        user_nlu_config_path: Path to user NLU configuration file.
        agent_nlu_config_path: Path to agent NLU configuration file.
        cached_user_nlu: Previously loaded user NLU module.
        cached_agent_nlu: Previously loaded agent NLU module.

    Returns:
        Tuple of (user_nlu, agent_nlu) modules.
    """
    if cached_user_nlu is None:
        user_nlu_config = Configuration("User NLU Configuration")
        user_nlu_config.set_file(user_nlu_config_path)
        cached_user_nlu = get_NLU(user_nlu_config)
    if cached_agent_nlu is None:
        agent_nlu_config = Configuration("Agent NLU Configuration")
        agent_nlu_config.set_file(agent_nlu_config_path)
        cached_agent_nlu = get_NLU(agent_nlu_config)
    return cached_user_nlu, cached_agent_nlu


def get_intent_lists(
    **kwargs: Any,
) -> Tuple[List[Intent], List[Intent], List[Intent]]:
    """Builds recommendation, acceptance, and rejection intent lists.

    Args:
        **kwargs: Optional intent label overrides:
            - recommendation_intent_labels: Labels for recommendation intents.
              Defaults to ``["REC-S", "REC-E"]``.
            - acceptance_intent_labels: Labels for acceptance intents.
              Defaults to ``["ACC"]``.
            - rejection_intent_labels: Labels for rejection intents.
              Defaults to ``["REJ"]``.

    Returns:
        Tuple of (recommendation_intents, acceptance_intents,
        rejection_intents).
    """
    rec_labels = kwargs.get("recommendation_intent_labels", ["REC-S", "REC-E"])
    acc_labels = kwargs.get("acceptance_intent_labels", ["ACC"])
    rej_labels = kwargs.get("rejection_intent_labels", ["REJ"])
    return (
        [Intent(label) for label in rec_labels],
        [Intent(label) for label in acc_labels],
        [Intent(label) for label in rej_labels],
    )


def annotate_dialogues(
    dialogues: List[Dialogue],
    user_nlu_config_path: str,
    agent_nlu_config_path: str,
) -> List[Dialogue]:
    """Annotates a batch of dialogues, loading NLU modules once.

    Args:
        dialogues: Dialogues to annotate.
        user_nlu_config_path: Path to user NLU configuration file.
        agent_nlu_config_path: Path to agent NLU configuration file.

    Returns:
        The same dialogue objects with annotated utterances.
    """
    user_nlu, agent_nlu = load_nlus(user_nlu_config_path, agent_nlu_config_path)
    for dialogue in dialogues:
        annotate_dialogue(dialogue, user_nlu, agent_nlu)
    return dialogues


def get_recommendation_rounds(
    dialogue: Dialogue, recommendation_intents: List[Intent]
) -> List[List[AnnotatedUtterance]]:
    """Splits a dialogue into recommendation rounds.

    A new round begins each time an utterance contains a recommendation
    intent.

    Args:
        dialogue: Annotated dialogue.
        recommendation_intents: Intents that signal a recommendation.

    Returns:
        List of utterance groups, one per recommendation round.
    """
    rounds: List[List[AnnotatedUtterance]] = []
    current_round: List[AnnotatedUtterance] = []
    for utterance in dialogue.utterances:
        if any(
            intent in utterance.get_intents()
            for intent in recommendation_intents
        ):
            if current_round:
                rounds.append(current_round)
            current_round = [utterance]
        else:
            current_round.append(utterance)
    return rounds


def prepare_dialogue(
    dialogue: Dialogue,
    user_nlu_config_path: str,
    agent_nlu_config_path: str,
    cached_user_nlu: Optional[NLU] = None,
    cached_agent_nlu: Optional[NLU] = None,
    **kwargs: Any,
) -> Tuple[Dialogue, List[Intent], List[Intent], List[Intent], NLU, NLU]:
    """Loads NLU modules, annotates a dialogue, and builds intent lists.

    Combines :func:`load_nlus`, :func:`annotate_dialogue`, and
    :func:`get_intent_lists` into a single convenience call.

    Args:
        dialogue: Dialogue to prepare.
        user_nlu_config_path: Path to user NLU configuration file.
        agent_nlu_config_path: Path to agent NLU configuration file.
        cached_user_nlu: Previously loaded user NLU module (avoids reload).
        cached_agent_nlu: Previously loaded agent NLU module (avoids reload).
        **kwargs: Optional intent label overrides forwarded to
            :func:`get_intent_lists`.

    Returns:
        Tuple of (annotated dialogue, recommendation intents,
        acceptance intents, rejection intents, user NLU, agent NLU).
    """
    user_nlu, agent_nlu = load_nlus(
        user_nlu_config_path,
        agent_nlu_config_path,
        cached_user_nlu,
        cached_agent_nlu,
    )
    annotate_dialogue(dialogue, user_nlu, agent_nlu)
    rec, acc, rej = get_intent_lists(**kwargs)
    return dialogue, rec, acc, rej, user_nlu, agent_nlu


def is_recommendation_accepted(
    round_utterances: List[AnnotatedUtterance],
    acceptance_intents: List[Intent],
    rejection_intents: List[Intent],
) -> bool:
    """Assesses whether a recommendation round was accepted.

    Args:
        round_utterances: Utterances in the recommendation round.
        acceptance_intents: Intents corresponding to acceptance.
        rejection_intents: Intents corresponding to rejection.

    Returns:
        True if the recommendation was accepted, False otherwise.
    """
    b_accepted = False
    for utterance in round_utterances:
        if utterance.participant == DialogueParticipant.USER:
            intents = utterance.get_intents()
            if any(intent in acceptance_intents for intent in intents):
                b_accepted = True
            elif any(intent in rejection_intents for intent in intents):
                return False
    return b_accepted
