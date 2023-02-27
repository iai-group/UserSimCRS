"""Represents a collection of items."""

import csv
from typing import Any, Dict

from dialoguekit.core.domain import Domain

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
        self,
        file_path: str,
        delimiter: str = ",",
        domain: Domain = None,
        domain_mapping: Dict[str, Dict[str, Any]] = {"ID": {"slot": "ID"}},
    ) -> None:
        """Loads an item collection from a CSV file.

        If items are connected to a Domain, only domain properties will be kept.

        Args:
            file_path: Path to CSV file.
            id_col: Name of the field containing item id. Defaults to 'ID'.
            name_col: Name of the field containing item name. Defaults to
              'NAME'.
            delimiter: Field separator, Defaults to ','.
            domain: Domain of the items. Defaults to None.
            domain_mapping: Field mapping to create item based on domain slots.

        Raises:
            ValueError: if there is no id column.
        """
        with open(file_path, "r", encoding="utf-8") as csvfile:
            csvreader = csv.DictReader(csvfile, delimiter=delimiter)
            for row in csvreader:
                properties = {}
                for field, _mapping in domain_mapping.items():
                    if _mapping["slot"] == "ID":
                        item_id = row.pop(field, None)
                        continue
                    properties[_mapping["slot"]] = (
                        row[field]
                        if not _mapping.get("multi-valued", False)
                        else row[field].split(_mapping["delimiter"])
                    )

                if not item_id:  # Checks if both ID and name exist.
                    raise ValueError(
                        "Item ID is mandatory. Please check that the correct "
                        "field mapping is used."
                    )

                item = Item(str(item_id), properties, domain)
                self.add_item(item)
