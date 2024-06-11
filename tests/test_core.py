import pytest
from chem_nlp.tokenizers import entity
from chem_nlp.core import ChemDoc, Settings
import os
import logging
from chem_nlp.tokenizers import vocab, tokenize
import re

logging.basicConfig(level=logging.DEBUG)

settings = Settings(
    ignore_case=True,
    token_merge_vocabs=[
        vocab.V_COMPOUND,
        vocab.V_COMPOUND_SYNONYM,
        vocab.V_FOOD,
        vocab.V_UNIT_WEIGHT,
    ],
    token_split_patterns=[tokenize.TM_UNITS],
    entity_patterns=[
        entity.EM_COMPOUND_SYN,
        entity.EM_COMPOUND,
        entity.EM_FOOD,
        entity.EM_QUALIFIER,
        entity.EM_MODIFIER,
        entity.EM_WEIGHT_VALUE,
        entity.EM_WEIGHT_UNIT,
    ],
    targets_per_sentence=2,
)


def load_test_data(filename):
    filepath = os.path.join(os.path.dirname(__file__), "test_data", filename)
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()


@pytest.mark.parametrize("filename,expected", [("text_sample_1.txt", ())])
def test_tokens(filename: str, expected):

    text = load_test_data(filename)
    doc = ChemDoc(text=text, settings=settings)

    compounds = []
    weight_units = []
    weight_values = []
    for sent in doc.sentences:
        for token in sent.tokens:
            logging.debug(f"{token.char}, {token.entity}, {token.pos}")
            # if token.entity in ["COMPOUND", "COMPOUND_SYN"]:
            #    compounds.append(token)
            # elif token.entity == "WEIGHT_UNIT":
            #    weight_units.append(token)
            # elif token.entity == "WEIGHT_VALUE":
            #    weight_values.append(token)

    # assert len(compounds) == expected[0]
    # assert len(weight_units) == expected[1]
    # assert len(weight_values) == expected[2]
