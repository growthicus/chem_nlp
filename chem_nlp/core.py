from __future__ import annotations

from typing import Callable
from chem_nlp.sentenzisers import sentenzise
from chem_nlp.tokenizers import tokenize
import nltk
import logging
from typing import Union

logging.basicConfig(level=logging.DEBUG)

# Download necessary NLTK data files
nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")


class Doc:

    def __init__(self, text: str, sentenizer: Callable = sentenzise.new_lines):
        self.text = text
        self.sentenizer = sentenizer
        self.sentences = self.sentenize()

    def sentenize(self) -> list[Sentence]:
        sentences = []
        for sent in self.sentenizer(self.text):
            sentences.append(Sentence(sent=sent, doc=self))

        return sentences


class Sentence:

    def __init__(self, doc: Doc, sent: str):
        self.doc = doc
        self.sent = sent
        self.initial_tokens = nltk.word_tokenize(sent)
        self.pos_tags = self.set_pos_tags(self.initial_tokens)
        self.tokens = self.tokenize()

    def is_first(self):
        return 0 == self.doc.sentences.index(self)

    def get_pos_tag(self, token: Token) -> str:
        # This can return the wrong POS if
        # the word occour more than once
        # and has differnt POS for each
        # occourence
        for word, pos in self.pos_tags:
            if word == token.char:
                return pos

    def set_pos_tags(self, tokens: list[str]):
        # Get the POS tags
        pos_tags = nltk.pos_tag(tokens)
        return pos_tags

    def tokenize(self) -> list[Token]:
        # tokens = []
        phrase_matches = tokenize.by_phrase_match(
            tokens=(self.initial_tokens), vocab=["vitamin a", "vitamin b", "calcium"]
        )

        # logging.info(phrase_matches)

        """
        for char in tokenize.by_phrase_match(
            tokens=(self.initial_tokens), vocab=["vitamin a", "vitamin b"]
        ):
            tokens.append(Token(sentence=self, char=char))

        # Re-init pos since merged can have change the tokens list
        self.pos_tags = self.set_pos_tags([token.char for token in tokens])
        return tokens
        """


class Token:

    def __init__(self, sentence: Sentence, char: str):
        self.sentence = sentence
        self.char = char

    def get_pos(self) -> str:
        return self.sentence.get_pos_tag(self)
