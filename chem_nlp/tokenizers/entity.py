from __future__ import annotations
import logging
from chem_nlp.dc import EntityMatcher
from chem_nlp.tokenizers import vocab

logging.basicConfig(level=logging.DEBUG)

EM_WEIGHT = EntityMatcher(
    name="WEIGHT",
    entity="WEIGHT",
    pattern_funcs=[
        lambda t: t.entity in ["COMPOUND", "COMPOUND_SYN"],
        lambda t: t.char in ["g", "mg"],
        # lambda t: True,
    ],
)


EM_COMPOUND = EntityMatcher(
    name="COMPOUND",
    entity="COMPOUND",
    pattern_funcs=[
        lambda t: t.char in vocab.V_COMPOUND.data,
        # lambda t: True,
    ],
)

EM_COMPOUND_SYN = EntityMatcher(
    name="COMPOUND_SYN",
    entity="COMPOUND_SYN",
    pattern_funcs=[
        lambda t: t.char in vocab.V_COMPOUND_SYNONYM.data,
        # lambda t: True,
    ],
)


def match_token_pattern(tokens: list, entity_patterns: list[EntityMatcher]):

    n = len(tokens)
    for entity_pattern in entity_patterns:

        m = len(entity_pattern.pattern_funcs)

        # Iterate over each string in the list
        for i in range(n):
            if i + m > n:  # Not enough strings left to match all functions
                break

            all_match = True
            # Check the sequence of functions
            for j in range(m):
                if not entity_pattern.pattern_funcs[j](tokens[i + j]):
                    all_match = False
                    break

            if all_match:
                tokens[i + j].entity = entity_pattern.entity

    return []


def by_token_pattern(tokens: list, entity_patterns: list[EntityMatcher]):

    return match_token_pattern(tokens=tokens, entity_patterns=entity_patterns)
