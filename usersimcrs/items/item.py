"""Represents an item."""

from typing import Any, Dict

from dialoguekit.core.domain import Domain


class Item:
    def __init__(
        self,
        item_id: str,
        name: str,
        properties: Dict[str, Any] = None,
        domain: Domain = None,
    ) -> None:
        """Create recsys item.

        Creates an item, which minimally has an ID and a canonical name, and
        can optionally have any number of properties, which are represented as
        key-value pairs.

        Args:
            item_id: Item ID.
            name: Item name.
            properties: Dictionary of item properties (key-value pairs).
              Defaults to None.
            domain: Domain knowledge. Defaults to None.
        """
        self._item_id = item_id
        self._name = name
        self._domain = domain

        if self._domain:
            self._properties = dict(
                filter(
                    lambda i: i[0] in self._domain.get_slot_names(),
                    properties.items(),
                )
            )
        else:
            self._properties = properties

    @property
    def id(self) -> str:
        """Return the item id."""
        return self._item_id

    @property
    def name(self) -> str:
        """Return the item name."""
        return self._name

    def get_property(self, key: str) -> Any:
        """Returns a given item property.

        Args:
            key: Name of property.

        Returns:
            Value of property or None.
        """
        return self._properties.get(key)

    def set_property(self, key: str, value: Any) -> None:
        """Sets the value of a given item property.

        If the item property exists it will be overwritten.

        Args:
            key: Property name.
            value: Property value.

        Raises:
            ValueError: if the property is not part of the domain knowledge.
        """
        if self._domain and key not in self._domain.get_slot_names():
            raise ValueError(
                f"The property {key} is not part of the Domain classes."
            )
        self._properties[key] = value
