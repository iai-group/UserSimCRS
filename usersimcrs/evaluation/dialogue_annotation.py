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
    one. Utterances that already carry dialogue acts are left untouched.

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


def load_nlu(
    nlu_config_path: str,
    config_name: str = "NLU Configuration",
    cached_nlu: Optional[NLU] = None,
) -> NLU:
    """Loads a single NLU module.

    Returns the cached instance when provided, otherwise creates a new one
    from the given configuration file.

    Args:
        nlu_config_path: Path to the NLU configuration file.
        config_name: Name for the Configuration instance.
        cached_nlu: Previously loaded NLU module.

    Returns:
        NLU module.
    """
    if cached_nlu is not None:
        return cached_nlu
    nlu_config = Configuration(config_name)
    nlu_config.set_file(nlu_config_path)
    return get_NLU(nlu_config)


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
) -> None:
    """Annotates a batch of dialogues in place, loading NLU modules once.

    Args:
        dialogues: Dialogues to annotate (modified in place).
        user_nlu_config_path: Path to user NLU configuration file.
        agent_nlu_config_path: Path to agent NLU configuration file.
    """
    user_nlu = load_nlu(user_nlu_config_path, "User NLU Configuration")
    agent_nlu = load_nlu(agent_nlu_config_path, "Agent NLU Configuration")
    for dialogue in dialogues:
        annotate_dialogue(dialogue, user_nlu, agent_nlu)


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
