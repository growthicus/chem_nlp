import pytest
from chem_nlp.core import Doc
import os
import logging

logging.basicConfig(level=logging.DEBUG)


def load_test_data(filename):
    filepath = os.path.join(os.path.dirname(__file__), "test_data", filename)
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()


@pytest.mark.parametrize("filename,expected", [("text_sample_1.txt", 8)])
def test_sentences(filename: str, expected):

    text = load_test_data(filename)
    doc = Doc(text=text)

    sent = doc.sentences[0]
    assert sent.is_first()


@pytest.mark.parametrize("filename,expected", [("text_sample_1.txt", 8)])
def test_tokens(filename: str, expected):

    text = load_test_data(filename)
    doc = Doc(text=text)
