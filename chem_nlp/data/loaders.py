import json
import os
from typing import List, Dict, Union
import logging
import re
import csv

logging.basicConfig(level=logging.DEBUG)


def filter_data(data: list[Dict], filter_rules: Dict[str, List[str]]):

    filtered_data = []

    for item in data:
        for filter_key, filters in filter_rules.items():
            if item[filter_key] not in filters:
                filtered_data.append(item)

    return filtered_data


def filter_replace_pattern(data: list[Dict], replace_rules: Dict[str, List[str]]):

    filtered_data = []

    for item in data:
        for replace_key, filters in replace_rules.items():
            new_item = item[replace_key]
            for filter in filters:
                new_item = new_item.replace(filter[0], filter[1])

            filtered_data.append(new_item)

    return filtered_data


def filter_data_with_regex(data, filter_rules):

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


def apply_filter(
    filename, data, keys_to_keep, only_values, to_lower, ignore, ignore_contains
):
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
        data = set(value for item in data for value in item.values())

    logging.info(f"{filename} final rows: {len(data)}")

    if not data:
        raise ValueError(f"No data loaded from {filename}")
    return data


def load_csv(
    filename: str,
    folder: str,
    keys_to_keep: List[str] = None,
    only_values: bool = False,
    to_lower: bool = False,
    ignore: Union[Dict[str, List[str]], None] = None,
    ignore_contains: Union[Dict[str, List[str]], None] = None,
):
    logging.info(f"loading file: {filename}")
    file_path = os.path.join(folder, filename)
    data = []

    with open(file_path, mode="r", newline="", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        for row in reader:
            if keys_to_keep:
                filtered_row = {key: row[key] for key in keys_to_keep if key in row}
            else:
                filtered_row = row

            data.append(filtered_row)

    return apply_filter(
        filename=filename,
        data=data,
        keys_to_keep=keys_to_keep,
        only_values=only_values,
        to_lower=to_lower,
        ignore=ignore,
        ignore_contains=ignore_contains,
    )


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
    with open(file_path, "r", encoding="utf-8") as file:
        data: List[Dict] = json.load(file)

    return apply_filter(
        filename=filename,
        data=data,
        keys_to_keep=keys_to_keep,
        only_values=only_values,
        to_lower=to_lower,
        ignore=ignore,
        ignore_contains=ignore_contains,
    )
