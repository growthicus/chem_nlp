import pytest
from typing import Callable, Tuple
from chem_nlp.settings import settings
from chem_nlp.sentenzisers import sentenzise
from chem_nlp.core import ChemDoc
import os
import logging

logging.basicConfig(level=logging.DEBUG)


def load_test_data(filename):
    filepath = os.path.join(os.path.dirname(__file__), "test_data", filename)
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()


@pytest.mark.parametrize(
    "filename, sentenziser, expected", [("text_sample_2.txt", sentenzise.standard, ())]
)
def test_tokens(filename: str, sentenziser: Callable, expected: Tuple[int, int, int]):

    text = load_test_data(filename)
    settings.sentenziser = sentenziser
    doc = ChemDoc(text=text, settings=settings)

    compounds = []
    weight_units = []
    weight_values = []
    for sent in doc.sentences:
        logging.error(f"############ {sent.sent}")
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
