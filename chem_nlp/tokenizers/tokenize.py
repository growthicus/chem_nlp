from __future__ import annotations
import logging
from nltk.util import bigrams, trigrams
from chem_nlp.dc import Matcher
from chem_nlp.tokenizers import vocab
import re

logging.basicConfig(level=logging.DEBUG)


TM_UNITS = Matcher(
    name="units",
    pattern_funcs=[
        lambda ct: re.match(
            rf"^([0-9]*\.?[0-9]+)\s*({'|'.join(vocab.V_UNIT_WEIGHT.load())})$",
            ct,
            flags=re.IGNORECASE,
        )
    ],
)


def match_word_sequence(
    char_tokens: list[str],
    vocab: list[str],
    max_sequence: int,
    max_match: int,
    min_n_gram_len: int,
) -> list[str]:

    if max_sequence == 2:
        gramify = bigrams
    elif max_sequence == 3:
        gramify = trigrams

    matches = []
    for i, x_gram in enumerate(list(gramify(char_tokens))):
        for i in range(len(x_gram)):
            for j in range(i + 1, len(x_gram) + 1):
                n_gram = " ".join(x_gram[i:j])
                if n_gram in vocab and len(n_gram) >= min_n_gram_len:
                    matches.append(n_gram)
                    latest_match = True
                    if len(matches) == max_match and len(x_gram) == max_sequence:
                        return matches

    return matches


def split_by_char_pattern(char_token: str, patterns: list[Matcher]) -> list[str]:

    char_tokens = []
    for pattern in patterns:
        for pattern_func in pattern.pattern_funcs:
            match = pattern_func(char_token)
            if match:
                char_tokens += [match.group(1), match.group(2)]
            else:
                char_tokens.append(char_token)

    return char_tokens


def merge_by_word_sqequence(
    char_tokens: list[str],
    vocab: list[str],
    max_sequence: int = 3,
    max_match: int = 1,
    min_n_gram_len: int = 3,
) -> list[str]:

    matches = match_word_sequence(
        char_tokens=char_tokens,
        vocab=vocab,
        max_sequence=max_sequence,
        max_match=max_match,
        min_n_gram_len=min_n_gram_len,
    )

    for match in sorted(matches, key=len):
        words_to_merge = match.split()
        length = len(words_to_merge)

        # Find the starting tokens of the sequence in the list
        for i in range(len(char_tokens) - length + 1):
            # Check if the subsequent elements in the list match the words to merge
            if char_tokens[i : i + length] == words_to_merge:
                # Replace the specific range with the new_value
                char_tokens[i : i + length] = [match]
                # Continue to next merge_dict without breaking to allow further replacements
                continue

    return char_tokens
