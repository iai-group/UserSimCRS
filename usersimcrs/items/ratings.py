"""Represents item ratings and provides access either based on items or users.

Ratings are normalized in the [-1,1] range, where -1 corresponds to
(strong) dislike, 0 is neutral, and 1 is (strong) like.
"""

import csv
import logging
import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from usersimcrs.items.item_collection import ItemCollection

logger = logging.getLogger(__name__)


def user_item_sampler(
    item_ratings: Dict[str, float],
    historical_ratio: float,
) -> List[str]:
    """Creates a random sample of items for a given user.

    Args:
        item_ratings: Item ratings to sample.
        historical_ratio: Ratio of items ratings to be used as historical
          data.

    Returns:
        List of sampled item ids.
    """
    # Determine the number of items to use as historical data for a given user.
    nb_historical_items = int(historical_ratio * len(item_ratings))
    return random.sample(item_ratings.keys(), nb_historical_items)


class Ratings:
    def __init__(self, item_collection: ItemCollection = None) -> None:
        """Initializes a ratings instance.

        Args:
            item_collection (optional): If provided, only ratings on items in
                ItemCollection are accepted.
        """
        self._item_collection = item_collection
        self._item_ratings: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._user_ratings: Dict[str, Dict[str, float]] = defaultdict(dict)

    def load_ratings_csv(
        self,
        file_path: str,
        delimiter: str = ",",
        min_rating: float = 0.5,
        max_rating: float = 5.0,
    ) -> None:
        """Loads ratings from a csv file.

        The file is assumed to have userID, itemID, and rating columns
        (following the MovieLens format). Additional columns that may be present
        are ignored. UserID and itemID are strings, rating is a float.

        Ratings are assumed to be given in the [min_rating, max_rating] range,
        which gets normalized into the [-1,1] range. (Default min/max rating
        values are based on the MovieLens collection.)

        If an ItemCollection is provided in the constructor, then ratings are
        filtered to items that are present in the collection.

        Args:
            file_path: Path to CSV file.
            delimiter: Field separator (default: comma).
            min_rating: Minimum rating (default: 0.5).
            max_rating: Maximum rating (default: 5.0).
        """
        with open(file_path, "r") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=delimiter)
            heading = next(csvreader)
            if len(heading) < 3:
                raise ValueError("Invalid CSV format (too few columns).")
            for values in csvreader:
                user_id, item_id = values[:2]
                rating = float(values[2])
                # Performs min-max normalization to [-1,1].
                normalized_rating = (
                    2 * (rating - min_rating) / (max_rating - min_rating) - 1
                )
                self.add_user_item_rating(user_id, item_id, normalized_rating)

    def get_user_ratings(self, user_id: str) -> Dict[str, float]:
        """Returns all ratings of a given user.

        Args:
            user_id: User ID.

        Returns:
            Dictionary with item IDs as keys and ratings as values.
        """
        return self._user_ratings[user_id]

    def get_item_ratings(self, item_id: str) -> Dict[str, float]:
        """Returns all ratings given to a specific item.

        Args:
            item_id: Item ID.

        Returns:
            Dictionary with user IDs as keys and ratings as values.
        """
        return self._item_ratings[item_id]

    def add_user_item_rating(
        self, user_id: str, item_id: str, normalized_rating: float
    ) -> None:
        """Adds the rating by a given user on a specific item.

        Args:
            user_id: User ID.
            item_id: Item ID.
            normalized_rating: Normalized rating.
        """
        # Filters items based on their existence in ItemCollection.
        if self._item_collection:
            if not self._item_collection.exists(item_id):
                logger.debug(f"Ratings for {item_id} are not included.")
                return
        self._item_ratings[item_id][user_id] = normalized_rating
        self._user_ratings[user_id][item_id] = normalized_rating

    def get_user_item_rating(
        self, user_id: str, item_id: str
    ) -> Optional[float]:
        """Returns the rating by a given user on a specific item.

        Args:
            user_id: User ID.
            item_id: Item ID.

        Returns:
            Rating as float or None.
        """
        return self._user_ratings[user_id].get(item_id, None)

    def get_random_user_id(self) -> str:
        """Returns a random user ID.

        Returns:
            User ID.
        """
        return random.choice(list(self._user_ratings.keys()))

    def create_split(
        self,
        historical_ratio: float,
        sampler: Callable = user_item_sampler,
    ) -> Tuple["Ratings", "Ratings"]:
        """Splits ratings into historical and ground truth ratings.

        Args:
            historical_ratio: Ratio ([0..1]) of ratings to be used as historical
              data.
            sampler: Callable performing the sampling of items per user.

        Raises:
            ValueError: if historical_ratio is not in the interval [0,1].

        Returns:
            Two Ratings objects, one corresponding to historical and another to
            ground truth ratings.
        """
        if historical_ratio > 1.0 or historical_ratio < 0.0:
            raise ValueError("historical_ratio is bounded in [0,1]")

        historical_ratings = Ratings(self._item_collection)
        ground_truth_ratings = Ratings(self._item_collection)

        for user_id, item_ratings in self._user_ratings.items():
            historical_item_ids = sampler(
                item_ratings,
                historical_ratio,
            )
            for item_id, rating in item_ratings.items():
                if item_id in historical_item_ids:
                    historical_ratings.add_user_item_rating(
                        user_id, item_id, rating
                    )
                else:
                    ground_truth_ratings.add_user_item_rating(
                        user_id, item_id, rating
                    )

        return historical_ratings, ground_truth_ratings
