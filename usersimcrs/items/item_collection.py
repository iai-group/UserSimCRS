"""Represents a collection of items.

Items are characterized by a set of properties, which correspond to slots in a
domain. Items are stored in a SQLite database and can be populated from a CSV
file.
"""

import csv
import sqlite3
from typing import Any, Dict, List, Set

from dialoguekit.core.annotation import Annotation

from usersimcrs.core.simulation_domain import SimulationDomain
from usersimcrs.items.item import Item

# Mapping configuration: for each csv field as key, it provides a dict with
# mapping instructions. This inner dict has minimally a "slot" key.
MappingConfig = Dict[str, Dict[str, Any]]
SQL_DELIMITER = "|"


class ItemCollection:
    def __init__(self, db_path: str, table_name: str) -> None:
        """Initializes an item collection.

        Args:
            db_path: Path to the SQLite database.
            table_name: Name of the table containing the items.
        """
        self._db_path = db_path
        self._table_name = table_name
        self.connect()

    def connect(self) -> None:
        """Connects to the SQLite database."""
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._cursor = self._conn.cursor()

    def close(self) -> None:
        """Closes the connection to the SQLite database.

        Connection and cursor are set to None to allow pickling.
        """
        self._conn.close()
        self._conn = None
        self._cursor = None

    def _init_table(self, domain_mapping: MappingConfig) -> None:
        """Initializes the table in the SQLite database.

        It is assumed that all the properties can be stored as TEXT.
        Multi-valued properties as stored as a string with the delimiter '|'.

        Args:
            domain_mapping: Mapping configuration.
        """
        properties = [v["slot"] for v in domain_mapping.values()]
        query = f"""CREATE TABLE IF NOT EXISTS {self._table_name}(
            id TEXT PRIMARY KEY,
            {', '.join([
                f'{prop} TEXT, {prop}_multi_value BOOLEAN'
                for prop in properties
            ])}
        )"""
        self._cursor.execute(query)
        self._conn.commit()

    def _parse_item_row(self, row: sqlite3.Row) -> Item:
        """Parses a row from the item table into an Item object.

        Args:
            row: Row from the item table.

        Returns:
            Item.
        """
        item_id = row["id"]
        properties = {
            key: (
                row[key]
                if not row[f"{key}_multi_value"]
                else row[key].split(SQL_DELIMITER)
            )
            for key in row.keys()
            if key != "id" and not key.endswith("_multi_value")
        }
        return Item(item_id, properties)

    def get_item(self, item_id: str) -> Item:
        """Returns an item from the collection based on its ID.

        Args:
            item_id: Item ID.

        Returns:
            Item or None, if not found.
        """
        self._cursor.execute(
            f"SELECT * FROM {self._table_name} WHERE id = ?", (item_id,)
        )
        row = self._cursor.fetchone()
        if row is None:
            return None

        return self._parse_item_row(row)

    def get_random_item(self) -> Item:
        """Returns a random item from the collection.

        Returns:
            Random item.
        """
        self._cursor.execute(
            f"SELECT * FROM {self._table_name} ORDER BY RANDOM() LIMIT 1"
        )
        row = self._cursor.fetchone()
        return self._parse_item_row(row)

    def exists(self, item_id: str) -> bool:
        """Checks if a given item exists in the item collection.

        Args:
            item_id: Item ID.

        Returns:
            True if the item exists in the collection.
        """
        self._cursor.execute(
            f"SELECT COUNT(*) FROM {self._table_name} WHERE id = {item_id}"
        )
        count = self._cursor.fetchone()[0]
        return count > 0

    def num_items(self) -> int:
        """Returns the number of items in the collection.

        Returns:
            Number of items.
        """
        self._cursor.execute(f"SELECT COUNT(*) FROM {self._table_name}")
        return self._cursor.fetchone()[0]

    def add_item(self, item: Item) -> None:
        """Adds an item to the collection.

        Args:
            item: Item.
        """
        properties = dict()
        for key, value in item.properties.items():
            if isinstance(value, list):
                properties[
                    key
                ] = f"""'{SQL_DELIMITER.join(value).replace("'","''")}'"""
                properties[f"{key}_multi_value"] = "True"
            else:
                properties[key] = f"""'{str(value).replace("'","''")}'"""
                properties[f"{key}_multi_value"] = "False"

        query = f"""REPLACE INTO {self._table_name}(id,
        {','.join(properties.keys())}) VALUES ('{item.id}',
        {','.join(list(properties.values()))});
        """

        self._cursor.execute(query)
        self._conn.commit()

    def load_items_csv(
        self,
        file_path: str,
        domain: SimulationDomain,
        domain_mapping: MappingConfig,
        id_col: str = "ID",
        delimiter: str = ",",
    ) -> None:
        """Loads an item collection from a CSV file.

        If items are connected to a Domain, only domain properties will be kept.

        Args:
            file_path: Path to CSV file.
            domain: Domain of the items.
            domain_mapping: Mapping configuration.
            id_col: Name of the column containing the item ID.
            delimiter: CSV delimiter.
        """
        self._init_table(domain_mapping)
        with open(file_path, "r", encoding="utf-8") as csvfile:
            csvreader = csv.DictReader(csvfile, delimiter=delimiter)
            for row in csvreader:
                item_id = row.pop(id_col, None)
                properties = {}

                for field, _mapping in domain_mapping.items():
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

    def get_possible_property_values(self, property: str) -> Set[str]:
        """Returns the possible values for a given property.

        Args:
            property: Property name.

        Returns:
            Set of possible values.
        """
        values: Set[str] = set()

        columns = [
            r["name"]
            for r in self._cursor.execute(
                f"PRAGMA table_info({self._table_name})"
            ).fetchall()
        ]
        if property not in columns:
            return values

        self._cursor.execute(
            f"SELECT DISTINCT {property}, {property}_multi_value "
            f"FROM {self._table_name}"
        )
        for row in self._cursor.fetchall():
            if row[1]:
                values.update(row[0].split(SQL_DELIMITER))
            else:
                values.add(row[0])
        return values

    def get_items_by_properties(
        self, annotations: List[Annotation]
    ) -> List[Item]:
        """Returns items that match the given annotations.

        Args:
            annotations: List of annotations.

        Returns:
            List of matching items.
        """
        matching_items: List[Item] = []

        if not annotations:
            return matching_items

        query = f"""SELECT * FROM {self._table_name} WHERE
        {' AND '.join([f"{a.slot} LIKE '%{a.value}%'" for a in annotations])}
        """
        print(query)
        self._cursor.execute(query)

        for row in self._cursor.fetchall():
            matching_items.append(self._parse_item_row(row))

        return matching_items
