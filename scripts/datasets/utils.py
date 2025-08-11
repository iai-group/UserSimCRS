"""Utility functions for transforming datasets into the DialogueKit format."""

from typing import Dict, Any, List
from itertools import chain

import pandas as pd

def merge_consecutive_utterances(
    utterances: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Merges consecutive utterances from the same participant.

    Args:
        utterances: List of utterances to merge.

    Returns:
        List of merged utterances.
    """
    merged_utterances: List[Dict[str, Any]] = []
    for utterance in utterances:
        last_utterance = merged_utterances[-1] if merged_utterances else {}

        if (
            not merged_utterances
            or utterance["participant"] != last_utterance["participant"]
        ):
            merged_utterances.append(utterance.copy())
        else:
            last_utterance["utterance"] += "\n" + utterance["utterance"]
            last_utterance["metadata"] = concatenate_metadata(
                last_utterance.get("metadata", {}),
                utterance.get("metadata", {})
            )
            last_utterance["annotations"] = concatenate_annotations(
                last_utterance.get("annotations", []),
                utterance.get("annotations", [])
            )
    return merged_utterances

def concatenate_metadata(metadata_utt1: Dict[str, Any], metadata_utt2: Dict[str, Any]) -> Dict[str, Any]:
    """Concatenates metadata from two utterances.

    Args:
        metadata_utt1: Metadata from the first utterance.
        metadata_utt2: Metadata from the second utterance.

    Returns:
        Combined metadata dictionary.
    """
    combined_metadata = metadata_utt1.copy()
    for key, value in metadata_utt2.items():
        if key in combined_metadata:
            combined_metadata[key] = _merge_values(combined_metadata[key], value)
        else:
            combined_metadata[key] = value
    return combined_metadata


def concatenate_annotations(
    annotations_utt1: List[Dict[str, Any]], annotations_utt2: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Concatenates annotations from two utterances.

    Args:
        annotations_utt1: Annotations from the first utterance.
        annotations_utt2: Annotations from the second utterance.

    Returns:
        Combined list of annotations.
    """
    combined_annotations = list(chain(annotations_utt1, annotations_utt2))
    combined_annotations_unique = list({frozenset((k, tuple(v) if isinstance(v, list) else v) for k, v in annotation.items()): annotation for annotation in combined_annotations})
    return [dict(ann) for ann in combined_annotations_unique]


def _merge_values(existing_value: Any, value_to_merge: Any) -> Any:
    """Merges two values into a single value.
    
    Args:
        existing_value: Current value.
        value_to_merge: Value to merge with the current value.

    Returns:
        Merged value.
    """
    if isinstance(existing_value, list):
        if isinstance(value_to_merge, list):
            existing_value.extend(value_to_merge)
        else:
            existing_value.append(value_to_merge)
        return existing_value
    
    if isinstance(value_to_merge, list):
        return [existing_value] + value_to_merge
    
    if isinstance(existing_value, dict) and isinstance(value_to_merge, dict):
        merged_dict = existing_value.copy()
        for key, value in value_to_merge.items():
            if key in merged_dict:
                merged_dict[key] = _merge_values(merged_dict[key], value)
            else:
                merged_dict[key] = value
        return merged_dict
    
    if isinstance(existing_value, str) and isinstance(value_to_merge, str):
        return f"{existing_value}[SEP]{value_to_merge}"
    
    if not existing_value and not value_to_merge:
        return None
    elif not existing_value:
        return value_to_merge
    elif not value_to_merge:
        return existing_value
    
    return [existing_value, value_to_merge]