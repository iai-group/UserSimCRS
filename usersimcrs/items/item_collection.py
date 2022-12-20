"""Represents a collection of items."""

import csv
from typing import Any, Dict, List

from usersimcrs.items.item import Item


class ItemCollection:
    def __init__(self) -> None:
        """Initializes an empty item collection."""
        self._items: Dict[str, Any] = {}

    def get_item(self, item_id: str) -> Item:
        """Returns an item from the collection based on its ID.

        Args:
            item_id: Item ID.

        Returns:
            Item or None, if not found.
        """
        return self._items.get(item_id)

    def exists(self, item_id: str) -> bool:
        """Checks if a given item exists in the item collection.

        Args:
            item_id: Item ID.

        Returns:
            True if the item exists in the collection.
        """
        return item_id in self._items

    def num_items(self) -> int:
        """Returns the number of items in the collection.

        Returns:
            Number of items.
        """
        return len(self._items)

    def add_item(self, item: Item) -> None:
        """Adds an item to the collection.

        Args:
            item: Item.
        """
        self._items[item.id] = item

    def load_items_csv(
        self, file_path: str, fields: List[Any], delimiter: str = ","
    ) -> None:
        """Loads an item collection from a CSV file.

        The CSV file is assumed to have a header row, which is ignored. Instead,
        the fields argument specifies which item properties the values
        correspond to.

        Args:
            file_path: Path to CSV file.
            fields: Mapping of CSV fields to item properties. ID and NAME are
                designated values for mapping to item ID and name. A None value
                means that the field is ignored. Otherwise, the value if fields
                is used as the name (key) of the item property.
            delimiter: Field separator (default: comma).
        """
        # TODO Optionally, connect items to an Domain and allow only for
        # properties that correspond to Domain classes.
        # See: https://github.com/iai-group/dialoguekit/issues/42
        with open(file_path, "r", encoding="utf-8") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=delimiter)
            next(csvreader)  # Skips header row.
            for values in csvreader:
                if len(fields) != len(values):
                    raise ValueError(
                        "Mismatch between provided fields and values in CSV "
                        "file."
                    )
                item_id = None
                name = None
                properties = {}
                for field, value in zip(fields, values):
                    if field == "ID":
                        item_id = value
                    elif field == "NAME":
                        name = value
                    elif field:
                        properties[field] = value

                if not (item_id and name):  # Checks if both ID and name exist.
                    raise ValueError("Item ID and Name are mandatory.")
                item = Item(item_id, name, properties)
                self.add_item(item)
