from parse_data import validate_label, load_train_data, load_validation_data
import logging

logging.basicConfig(level=logging.ERROR)

for i, entry in enumerate(load_train_data()):
    if entry["labels"] == "":
        continue

    for label in entry["labels"].split(","):
        try:
            validate_label(label)
        except ValueError as e:
            logging.error(f"train row: {i} error:{e}")


for i, entry in enumerate(load_validation_data()):
    if entry["labels"] == "":
        continue

    for label in entry["labels"].split(","):
        try:
            validate_label(label)
        except ValueError as e:
            logging.error(f"validation row: {i} error:{e}")
