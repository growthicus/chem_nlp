import json
import os
from typing import List, Dict, Union
import logging
import re

logging.basicConfig(level=logging.DEBUG)


def filter_data(data: list[Dict], filter_rules: Dict[str, List[str]]):

    filtered_data = []

    for item in data:
        for filter_key, filters in filter_rules.items():
            try:
                if item[filter_key] not in filters:
                    filtered_data.append(item)
            except KeyError:
                raise KeyError(f"Cannot find key {filter_key} in {item}")
    return filtered_data


def filter_data_with_regex(data, filter_rules):
    """
    Filters a list of dictionaries based on exclusion regex rules defined in filter_rules.

    :param data: List of dictionaries to filter
    :param filter_rules: Dictionary specifying regex patterns to exclude for each key
    :return: The modified list of dictionaries with values matching the regex patterns removed
    """

    # Create a new list to store the filtered dictionaries
    filtered_data = []

    # Iterate through each dictionary in the data list
    for item in data:
        # Create a new dictionary to store the filtered items
        filtered_item = {}

        # Check each key-value pair in the dictionary
        for key, value in item.items():
            # If the key is in the filter rules and the value does not match any regex pattern, add it to the filtered dictionary
            if key not in filter_rules or not any(
                re.search(pattern, value) for pattern in filter_rules[key]
            ):
                filtered_item[key] = value

        # Add the filtered dictionary to the filtered data list
        filtered_data.append(filtered_item)

    return filtered_data


def load_json(
    filename: str,
    folder: str,
    keys_to_keep: List[str] = None,
    only_values: bool = False,
    to_lower: bool = False,
    ignore: Union[Dict[str, List[str]], None] = None,
    ignore_contains: Union[Dict[str, List[str]], None] = None,
):
    file_path = os.path.join(folder, filename)

    logging.info(f"loading file: {filename}")
    with open(file_path, "r") as file:
        data: List[Dict] = json.load(file)

    logging.info(f"{filename} initial rows: {len(data)}")

    # Apply filtering based on ignore lists
    if ignore:
        data = filter_data(data, ignore)

    if ignore_contains:
        data = filter_data_with_regex(data, ignore_contains)

    if keys_to_keep is not None:
        data = [
            {key: item[key] for key in keys_to_keep if key in item} for item in data
        ]

    if to_lower:
        data = [
            {k: v.lower() if isinstance(v, str) else v for k, v in item.items()}
            for item in data
        ]

    if only_values:
        data = [value for item in data for value in item.values()]

    logging.info(f"{filename} final rows: {len(data)}")
    return data
