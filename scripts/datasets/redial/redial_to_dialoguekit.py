"""Script to format the ReDial dataset into DialogueKit format.

This script downloads the ReDial dataset and formats the dialogues following
DialogueKit's format. Formatting includes merging consecutive utterances from
the same participant and replacing movie mentions with their titles and years.
In addition to formatting the dialogues, this script also processes items and
ratings information.

Reference:
Li, Raymond, et al. "Towards deep conversational recommendations." Advances in
neural information processing systems 31 (2018).
"""

import json
import logging
import os
import re
import zipfile
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import pandas as pd
import wget

from scripts.datasets.utils import merge_consecutive_utterances

REDIAL_DATASET_URL = (
    "https://github.com/ReDialData/website/raw/data/redial_dataset.zip"
)
DATASET_PATH = "data/datasets/redial"
ITEM_COLLECTION_PATH = "data/item_collections/redial"

Items = Dict[str, Dict[str, Any]]
Rating = Tuple[str, str, float]

logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def download_redial_dataset(
    dataset_url: str = REDIAL_DATASET_URL, dataset_path: str = DATASET_PATH
) -> None:
    """Downloads the ReDial dataset from the official source.

    Args:
        dataset_url: URL of the ReDial dataset.
        dataset_path: Path to save the ReDial dataset.
    """
    wget.download(dataset_url, out="scripts/redial_dataset.zip")

    with zipfile.ZipFile("scripts/redial_dataset.zip", "r") as zip_folder:
        zip_folder.extractall(dataset_path)

    os.remove("scripts/redial_dataset.zip")


def load_data(
    path: str,
    items: Items = None,
    ratings: List[Rating] = None,
) -> Tuple[List[Dict[str, Any]], Items, List[Rating]]:
    """Loads dialogues from a JSONL file.

    Args:
        path: Path to the JSONL file.
        items: Items mentioned in the dialogues.
        ratings: Ratings extracted from the dialogues.

    Returns:
        List of dialogues, items mentioned in the dialogues, and ratings.
    """
    data = []
    items = deepcopy(items) if items else dict()
    ratings = deepcopy(ratings) if ratings else list()

    for line in open(path, "r"):
        dialogue, _items, _ratings = _format_dialogue(json.loads(line))
        data.append(dialogue)
        items.update(_items)
        ratings.extend(_ratings)
    return data, items, ratings


def _format_dialogue(
    dialogue: Dict[str, Any]
) -> Tuple[Dict[str, Any], Items, List[Rating]]:
    """Formats a dialogue into DialogueKit format.

    Args:
        dialogue: Dialogue to format.

    Returns:
        Formatted dialogue, items mentioned in the dialogue, and ratings.
    """
    formatted_dialogue = {
        "conversation_id": dialogue["conversationId"],
        "agent": {
            "id": dialogue["respondentWorkerId"],
            "type": "AGENT",
        },
        "user": {
            "id": dialogue["initiatorWorkerId"],
            "type": "USER",
        },
    }
    items = _get_items(dialogue)
    ratings = _get_ratings(dialogue, dialogue["initiatorWorkerId"])
    utterances = _parse_utterances(
        dialogue, dialogue["initiatorWorkerId"], items
    )
    formatted_dialogue["conversation"] = utterances
    return formatted_dialogue, items, ratings


def _parse_utterances(
    dialogue: Dict[str, Any], user_id: str, items: Items
) -> List[Dict[str, Any]]:
    """Parses utterances from the dialogue.

    Args:
        dialogue: Dialogue to parse utterances from.
        user_id: ID of the user in the dialogue.
        items: Items mentioned in the dialogue.

    Returns:
        List of parsed utterances.
    """
    utterances = []
    for message in dialogue["messages"]:
        participant = (
            "USER" if message["senderWorkerId"] == user_id else "AGENT"
        )
        text = message["text"]
        utterance = {
            "participant": participant,
            "metadata": {
                "timeOffset": message["timeOffset"],
            },
        }

        pattern = r"(@\d+)"
        matches = re.findall(pattern, text)
        if matches:
            for m in matches:
                movie_id = m[1:]
                movie_title = items.get(movie_id, {}).get("title", None)
                if movie_title:
                    movie_year = items.get(movie_id, {}).get("year", None)
                    movie = (
                        f"{movie_title} ({movie_year})"
                        if movie_year
                        else movie_title
                    )
                    text = text.replace(m, movie)

        utterance["utterance"] = text
        utterances.append(utterance)

    utterances = merge_consecutive_utterances(utterances)
    return utterances


def _get_items(dialogue: Dict[str, Any]) -> Items:
    """Extracts items mentioned in the dialogue.

    Args:
        dialogue: Dialogue to extract items from.

    Raises:
        AttributeError: If no movie mentions found.

    Returns:
        Items mentioned in the dialogue.
    """
    items = {}
    pattern = r"(?P<title>.+)\s\((?P<year>\d{4})\)"
    try:
        for id, value in dialogue["movieMentions"].items():
            if not value:
                continue
            match = re.search(pattern, value)
            if match:
                title = match.group("title").strip()
                year = int(match.group("year"))
                items[id] = {"title": title, "year": year}
    except AttributeError as e:
        # No movie mentions found.
        logger.error(e)

    return items


def _get_ratings(dialogue: Dict[str, Any], user_id: str) -> List[Rating]:
    """Extracts ratings from the dialogue.

    Args:
        dialogue: Dialogue to extract ratings from.
        user_id: ID of the user who provided the ratings.

    Raises:
        AttributeError: If no questions found.

    Returns:
        List of ratings extracted from the dialogue.
    """
    ratings = list()
    for q in ["initiatorQuestions", "respondentQuestions"]:
        try:
            for id, value in dialogue.get(q, {}).items():
                rating = value["liked"]
                if rating != 2:
                    ratings.append((user_id, id, rating))
        except AttributeError as e:
            # No questions found.
            logger.error(e)
    return ratings


if __name__ == "__main__":
    for p in [DATASET_PATH, ITEM_COLLECTION_PATH]:
        if not os.path.exists(p):
            os.makedirs(p, exist_ok=True)

    download_redial_dataset()

    items: Items = {}
    ratings: List[Rating] = []

    train_data, items, ratings = load_data(
        os.path.join(DATASET_PATH, "train_data.jsonl"), items, ratings
    )
    test_data, items, ratings = load_data(
        os.path.join(DATASET_PATH, "test_data.jsonl"), items, ratings
    )

    # Save formatted dialogues
    with open(os.path.join(DATASET_PATH, "train_dialogues.json"), "w") as f:
        json.dump(train_data, f, indent=4)
        logger.info(f"Saved {len(train_data)} training dialogues in: {f.name}")

    with open(os.path.join(DATASET_PATH, "test_dialogues.json"), "w") as f:
        json.dump(test_data, f, indent=4)
        logger.info(f"Saved {len(test_data)} test dialogues in: {f.name}")

    with open(os.path.join(DATASET_PATH, "all_dialogues.json"), "w") as f:
        json.dump(train_data + test_data, f, indent=4)

    # Save items
    items_df = pd.DataFrame(items).T.reset_index()
    items_df.columns = ["item_id", "title", "year"]
    items_df.drop_duplicates(subset=["item_id"], inplace=True)
    logger.info(f"Collected {len(items_df)} items.")
    items_df.to_csv(
        os.path.join(ITEM_COLLECTION_PATH, "movies.csv"), index=False
    )

    # Save ratings
    ratings_df = pd.DataFrame(
        ratings,
        columns=["user_id", "item_id", "rating"],
    )
    ratings_df.drop_duplicates(inplace=True)
    logger.info(f"Collected {len(ratings_df)} ratings.")
    ratings_df.to_csv(
        os.path.join(ITEM_COLLECTION_PATH, "ratings.csv"), index=False
    )
