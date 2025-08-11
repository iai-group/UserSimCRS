"""Script to format the INSPIRED dataset into DialogueKit format.

The script downloads the INSPIRED dataset, including the dialogues and the item collection. The dialogues are formatted to follow DialogueKit's structure, ...

Reference:
Hayati, Shirley Anugrah, et al. "INSPIRED: Toward Sociable Recommendation Dialog Systems." Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2020.
"""

import json
import logging
import numpy as np
import pandas as pd
import wget
import os
import ast
from typing import Dict, List, Tuple, Any

from scripts.datasets.utils import merge_consecutive_utterances

INSPIRED_DIALOGUES_URLS = [
    "https://raw.githubusercontent.com/sweetpeach/Inspired/refs/heads/master/data/dialog_data/train.tsv",
    "https://raw.githubusercontent.com/sweetpeach/Inspired/refs/heads/master/data/dialog_data/dev.tsv",
    "https://raw.githubusercontent.com/sweetpeach/Inspired/refs/heads/master/data/dialog_data/test.tsv",
]
INSPIRED_ITEMS_URL = "https://raw.githubusercontent.com/sweetpeach/Inspired/refs/heads/master/data/movie_database.tsv"  # video_id as item_id in dialogues
DATASET_PATH = "data/datasets/inspired"
ITEM_COLLECTION_PATH = "data/item_collections/inspired"

USER_ID = "SEEKER"
AGENT_ID = "RECOMMENDER"

Items = Dict[str, Dict[str, Any]]
Rating = Tuple[str, str, float]

logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def download_inspired_data(
    dataset_urls: List[str] = INSPIRED_DIALOGUES_URLS,
    dataset_path: str = DATASET_PATH,
    items_url: str = INSPIRED_ITEMS_URL,
    item_collection_path: str = ITEM_COLLECTION_PATH,
) -> None:
    """Downloads the INSPIRED data from the official source.

    Args:
        dataset_urls: List of URLs of the INSPIRED dialogues.
        dataset_path: Path to save the INSPIRED dialogues.
        items_url: URL of the INSPIRED items.
        item_collection_path: Path to save the INSPIRED items.
    """
    for dataset_url in dataset_urls:
        wget.download(dataset_url, out=dataset_path)

    wget.download(items_url, out=item_collection_path)

def process_data(data_path:str, items:pd.DataFrame) -> List[Dict[str, Any]]:
    """Processes the dataframe to format it into the DialogueKit format.
    
    Args:
        data_path: Path to the TSV file containing dialogues.
        items: DataFrame containing item information.
    
    Returns:
        List of formatted dialogues.
    """
    data = pd.read_csv(data_path, sep="\t")
    data[["movie_dict", "genre_dict", "actor_dict", "director_dict", "others_dict"]] = data[["movie_dict", "genre_dict", "actor_dict", "director_dict", "others_dict"]].apply(convert_to_dict)
    dialogues = []

    for _, group in data.groupby("dialog_id"):
        formatted_dialogue = format_dialogue(group, items)
        dialogues.append(formatted_dialogue)

    return dialogues

def convert_to_dict(values: pd.Series) -> pd.Series:
        """Converts cell value to a dictionary.
        
        Args:
            value: Cell value to convert.
            
        Returns:
            Series with converted values.
        """
        converted_values = []

        for value in values:
            try:
                if pd.isna(value):
                    converted_values.append({})
                else:
                    converted_values.append(ast.literal_eval(value))
            except (ValueError, SyntaxError) as e:
                logger.warning(f"Error converting value {value} to dict: {e}")
                converted_values.append(value)
        return pd.Series(converted_values)

def format_dialogue(
    dialogue: pd.DataFrame, items: pd.DataFrame
) -> Dict[str, Any]:
    """Formats a dialogue into the DialogueKit format.

    Args:
        dialogue: A DataFrame containing the dialogue data.
        items: A DataFrame containing the item data.

    Returns:
        Formatted dialogue as a JSON dictionary.
    """
    first_row_dict = dialogue.iloc[0].to_dict()
    
    formatted_dialogue = {
        "conversation_id": first_row_dict.get("dialog_id", None),
        "agent": {
            "id": AGENT_ID,
            "type": "AGENT",
        },
        "user": {
            "id": USER_ID,
            "type": "USER",
        },
        "metadata": {
            "movie_dict": first_row_dict.get("movie_dict"),
            "genre_dict": first_row_dict.get("genre_dict"),
            "actor_dict": first_row_dict.get("actor_dict"),
            "director_dict": first_row_dict.get("director_dict"),
            "other_names_dict": first_row_dict.get("others_dict"),
            "recommendation_outcome": first_row_dict.get("fine_label", None),
        },
    }

    if "movie_id" in first_row_dict:
        recommended_movie_title = items[items["video_id"] == first_row_dict["movie_id"]].iloc[0].get("title", None)
        formatted_dialogue["metadata"]["recommended_movie"] = recommended_movie_title

    utterances = parse_utterances(dialogue)
    formatted_dialogue["conversation"] = utterances
    return formatted_dialogue


def parse_utterances(dialogue: pd.DataFrame) -> List[Dict[str, Any]]:
    """Parses utterances from a dialogue DataFrame.

    Args:
        dialogue: A DataFrame containing the dialogue data.

    Returns:
        List of parsed utterances.
    """
    utterances = []

    for _, row in dialogue.iterrows():
        utterance = {
            "participant": "USER" if row["speaker"] == USER_ID else "AGENT",
            "utterance": row["text"].strip(),
            "metadata": {"utterance_with_placeholders": row["text_with_placeholder"].strip(),
            },
            "annotations": []
        }
        if not pd.isna(row["movies"]):
            movies_annotations = [m.strip() for m in row["movies"].split(";")]
            utterance["annotations"].append({
                "key": "movies",
                "value": movies_annotations,
            })

        if not pd.isna(row["genres"]):
            genres_annotations = [g.strip() for g in row["genres"].split(";")]
            utterance["annotations"].append({
                "key": "genres",
                "value": genres_annotations,
            })

        if not pd.isna(row["people_names"]):
            people_names_annotations = [p.strip() for p in row["people_names"].split(";")]
            utterance["annotations"].append({
                "key": "people_names",
                "value": people_names_annotations,
            })

        if not pd.isna(row["expert_label"]):
            utterance["annotations"].append({
                "key": "first_social_label",
                "value": row["expert_label"].strip(),
            })
            
        if not pd.isna(row["second_label"]):
            utterance["annotations"].append({
                "key": "second_social_label",
                "value": row["second_label"].strip(),
            })

        utterances.append(utterance)

    utterances = merge_consecutive_utterances(utterances)
    return utterances


if __name__ == "__main__":
    for p in [DATASET_PATH, ITEM_COLLECTION_PATH]:
        if not os.path.exists(p):
            os.makedirs(p, exist_ok=True)

    download_inspired_data()

    items = pd.read_csv(
        os.path.join(ITEM_COLLECTION_PATH, "movie_database.tsv"),
        sep="\t",
    )

    train_dialogues = process_data(
        os.path.join(DATASET_PATH, "train.tsv"), items
    )
    dev_dialogues = process_data(
        os.path.join(DATASET_PATH, "dev.tsv"), items
    )
    test_dialogues = process_data(
        os.path.join(DATASET_PATH, "test.tsv"), items
    )

    # Save formatted dialogues
    with open(os.path.join(DATASET_PATH, "train_dialogues.json"), "w") as f:
        json.dump(train_dialogues, f, indent=4)
        logger.info(f"Saved {len(train_dialogues)} training dialogues in: {f.name}")

    with open(os.path.join(DATASET_PATH, "dev_dialogues.json"), "w") as f:
        json.dump(dev_dialogues, f, indent=4)
        logger.info(f"Saved {len(dev_dialogues)} development dialogues in: {f.name}")

    with open(os.path.join(DATASET_PATH, "test_dialogues.json"), "w") as f:
        json.dump(test_dialogues, f, indent=4)
        logger.info(f"Saved {len(test_dialogues)} test dialogues in: {f.name}")

    with open(os.path.join(DATASET_PATH, "all_dialogues.json"), "w") as f:
        json.dump(train_dialogues + dev_dialogues + test_dialogues, f, indent=4)
        logger.info(f"Saved all dialogues in: {f.name}")

    logger.info("INSPIRED dataset processing completed.")
