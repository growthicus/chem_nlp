from chem_nlp.data.loaders import load_csv


def load_train_data():
    data = load_csv("train.csv", "models/ingredients/data", to_lower=True)
    return data


def load_validation_data():
    data = load_csv("validate.csv", "models/ingredients/data", to_lower=True)
    return data


def load_eval_data():
    data = load_csv("eval.csv", "models/ingredients/data", to_lower=True)
    return data


def validate_label(label):
    banned = [
        "{",
        "}",
        "[",
        "]",
        "(",
        ")",
        ":",
        ";",
        ",",
        ".",
        "<",
        ">",
        "?",
        "/",
        "\\",
        "|",
        "`",
        "~",
        "!",
        "@",
        "$",
        "%",
        "^",
        "*",
        "_",
        "+",
        "=",
        "and",
        "contains",
    ]
    # CHECK IF LABEL CONTAINS ANY BANNED CHARACTERS
    for char in banned:
        if char in label:
            raise ValueError(f"{label} contains banned character: {char}")

    if label == "":
        raise ValueError("Label cannot be empty")

    return label


def parse_training_data(data):
    parsed_data = []

    for entry in data:
        text = entry["ingredients"].lower()
        if not entry["labels"]:
            continue

        labels = [
            validate_label(label.lower()).strip()
            for label in entry["labels"].split(",")
        ]
        entities = []
        start = 0

        # Find the start and end indices of each label in the text
        for label in labels:
            start = text.find(label, start)
            if (
                start == -1
            ):  # If the label is not found in the text, continue to next label
                continue
            end = start + len(label)
            entities.append((start, end, "FOOD"))
            start = end  # Move start to end for next search, prevents finding the same label again

        parsed_data.append((text, {"entities": entities}))

    return parsed_data
