import pytest
from chem_nlp.tokenizers import entity
from chem_nlp.core import ChemDoc, Settings
import os
import logging
from chem_nlp.tokenizers import vocab

logging.basicConfig(level=logging.DEBUG)

settings = Settings(
    ignore_case=True,
    token_vocabs=[
        vocab.V_COMPOUND,
        vocab.V_COMPOUND_SYNONYM,
        vocab.V_FOOD,
        vocab.V_UNIT_WEIGHT,
    ],
    entity_patterns=[entity.EM_COMPOUND_SYN, entity.EM_COMPOUND, entity.EM_WEIGHT],
    targets_per_sentence=2,
)


def load_test_data(filename):
    filepath = os.path.join(os.path.dirname(__file__), "test_data", filename)
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()


@pytest.mark.parametrize("filename,expected", [("text_sample_1.txt", 8)])
def test_tokens(filename: str, expected):

    text = load_test_data(filename)
    doc = ChemDoc(text=text, settings=settings)

    for token in doc.sentences[0].tokens:
        logging.info(f"TOKEN: {token}")
