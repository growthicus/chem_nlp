from __future__ import annotations
import logging
from chem_nlp.dc import Matcher
from chem_nlp.tokenizers import vocab
from chem_nlp.tokenizers.tokenize import remove_nomenclature_postfix
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
    pattern_funcs=[
        (
            [
                lambda t, s: is_numeric(t.char),
                lambda t, s: lemmatize(t.char) in vocab.V_UNIT_WEIGHT.load(),
            ],
            0,
        ),
    ],
)


EM_WEIGHT_UNIT = Matcher(
    name="WEIGHT_UNIT",
    pattern_funcs=[
        (
            [
                lambda t, s: is_numeric(t.char),
                lambda t, s: lemmatize(t.char) in vocab.V_UNIT_WEIGHT.load(),
            ],
            1,
        )
    ],
)


EM_COMPOUND = Matcher(
    name="COMPOUND",
    pattern_funcs=[
        (
            [
                lambda t, s: remove_nomenclature_postfix(t.char)
                in vocab.V_COMPOUND.load()
                and len(t.char) >= s.min_vocab_word_length,
            ],
            0,
        )
    ],
)

EM_COMPOUND_SYN = Matcher(
    name="COMPOUND_SYN",
    pattern_funcs=[
        (
            [
                lambda t, s: remove_nomenclature_postfix(t.char)
                in vocab.V_COMPOUND_SYNONYM.load()
                and len(t.char) >= s.min_vocab_word_length,
            ],
            0,
        )
    ],
)

EM_FOOD = Matcher(
    name="FOOD",
    pattern_funcs=[
        (
            [
                lambda t, s: lemmatize(t.char) in vocab.V_FOOD.load()
                and len(t.char) >= s.min_vocab_word_length,
            ],
            0,
        ),
    ],
)

EM_MODIFIER = Matcher(
    name="MODIFIER",
    pattern_funcs=[
        (
            [
                lambda t, s: t.entity in ["FOOD", "COMPOUND", "COMPOUND_SYN"],
                lambda t, s: t.char in vocab.V_MODIFIER.load(),
            ],
            1,
        ),
        (
            [
                lambda t, s: t.entity in ["MODIFIER"],
                lambda t, s: t.char in vocab.V_MODIFIER.load(),
            ],
            1,
        ),
    ],
)


EM_QUALIFIER = Matcher(
    name="QUALIFIER",
    pattern_funcs=[
        (
            [
                lambda t, s: t.pos == "VBN",
                lambda t, s: t.entity in ["FOOD", "COMPOUND", "COMPOUND_SYN"],
            ],
            0,
        ),
    ],
)


def match_word_pattern(
    tokens: list, match_patterns: list[Matcher], settings, max_match: int = 3
):
    n = len(tokens)
    matches = []

    # Iterate over each entity pattern
    for match_pattern in match_patterns:
        logging.info(f"STARTED {match_pattern.name}")

        # Iterate over each pattern sequence in the Matcher object
        for pattern, index in match_pattern.pattern_funcs:
            m = len(pattern)
            # Iterate over each token in the list
            for i in range(n):
                if i + m > n:  # Not enough tokens left to match all functions
                    break

                all_match = True
                # Check the sequence of functions
                for j in range(m):
                    if not pattern[j](tokens[i + j], settings):
                        all_match = False
                        break

                # If all functions matched
                if all_match:
                    # Apply the entity to the specified token in the sequence
                    if i + index < n:
                        tokens[i + index].entity = match_pattern.name
                        matches.append((tokens[i + index], match_pattern))
                    else:
                        raise Exception("Index out of range for matcher result")

    return matches


def by_word_pattern(tokens: list, patterns: list[Matcher], settings, max_match: int):

    return match_word_pattern(
        tokens=tokens, match_patterns=patterns, settings=settings, max_match=max_match
    )
