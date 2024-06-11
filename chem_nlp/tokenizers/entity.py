from __future__ import annotations
import logging
from chem_nlp.dc import Matcher
from chem_nlp.tokenizers import vocab
from nltk.stem import WordNetLemmatizer

lemmatize = WordNetLemmatizer().lemmatize

logging.basicConfig(level=logging.DEBUG)


def is_numeric(value: str):
    try:
        return float(value)
    except ValueError:
        return False


EM_WEIGHT_VALUE = Matcher(
    name="WEIGHT_VALUE",
    entity="WEIGHT_VALUE",
    pattern_funcs=[
        (
            [
                lambda t: is_numeric(t.char),
                lambda t: lemmatize(t.char) in vocab.V_UNIT_WEIGHT.load(),
            ],
            0,
        ),
    ],
)


EM_WEIGHT_UNIT = Matcher(
    name="WEIGHT_UNIT",
    entity="WEIGHT_UNIT",
    pattern_funcs=[
        (
            [
                lambda t: is_numeric(t.char),
                lambda t: lemmatize(t.char) in vocab.V_UNIT_WEIGHT.load(),
            ],
            1,
        )
    ],
)


EM_COMPOUND = Matcher(
    name="COMPOUND",
    entity="COMPOUND",
    pattern_funcs=[
        (
            [
                lambda t: t.char in vocab.V_COMPOUND.load() and len(t.char) >= 3,
            ],
            0,
        )
    ],
)

EM_COMPOUND_SYN = Matcher(
    name="COMPOUND_SYN",
    entity="COMPOUND_SYN",
    pattern_funcs=[
        (
            [
                lambda t: t.char in vocab.V_COMPOUND_SYNONYM.load()
                and len(t.char) >= 3,
            ],
            0,
        )
    ],
)

EM_FOOD = Matcher(
    name="FOOD",
    entity="FOOD",
    pattern_funcs=[
        (
            [
                lambda t: t.char in vocab.V_FOOD.load() and len(t.char) >= 3,
            ],
            0,
        )
    ],
)

EM_MODIFIER = Matcher(
    name="MODIFIER",
    entity="MODIFIER",
    pattern_funcs=[
        (
            [
                lambda t: t.entity in ["FOOD", "COMPOUND", "COMPOUND_SYN"],
                lambda t: t.char in vocab.V_MODIFIER.load(),
            ],
            1,
        ),
    ],
)


EM_QUALIFIER = Matcher(
    name="QUALIFIER",
    entity="QUALIFIER",
    pattern_funcs=[
        (
            [
                lambda t: t.pos == "VBN",
                lambda t: t.entity in ["FOOD", "COMPOUND", "COMPOUND_SYN"],
            ],
            0,
        ),
    ],
)


def match_token_pattern(tokens: list, entity_patterns: list[Matcher]):
    n = len(tokens)

    # Iterate over each entity pattern
    for entity_pattern in entity_patterns:

        # Iterate over each pattern sequence in the Matcher object
        for pattern, entity_index in entity_pattern.pattern_funcs:

            m = len(pattern)

            # Iterate over each token in the list
            for i in range(n):
                if i + m > n:  # Not enough tokens left to match all functions
                    break

                all_match = True
                # Check the sequence of functions
                for j in range(m):
                    if not pattern[j](tokens[i + j]):
                        all_match = False
                        break

                # If all functions matched
                if all_match:
                    # Apply the entity to the specified token in the sequence
                    if (
                        i + entity_index < n
                    ):  # Ensure the index is within the range of the token list
                        tokens[i + entity_index].entity = entity_pattern.entity
                    break  # Break after finding a match to avoid overlapping entities

    return tokens  # Return the modified tokens


def by_token_pattern(tokens: list, entity_patterns: list[Matcher]):

    return match_token_pattern(tokens=tokens, entity_patterns=entity_patterns)
