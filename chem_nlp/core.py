from __future__ import annotations

from typing import Callable
from chem_nlp.sentenzisers import sentenzise
from chem_nlp.tokenizers import tokenize
from chem_nlp.data.loaders import load_json
import nltk
import logging
from typing import Union
from dataclasses import dataclass

logging.basicConfig(level=logging.DEBUG)

# Download necessary NLTK data files
nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")


@dataclass
class Settings:
    sentenziser: Callable = sentenzise.new_lines
    initial_tokenizer: Callable = nltk.word_tokenize
    tokenize_compounds: bool = True
    tokenize_foods: bool = False
    targets_per_sentence: int = 1
    max_sequence: int = 3


class ChemDoc:

    def __init__(self, text: str, settings: Settings):
        self.text = text
        self.settings = settings
        self.vocab = self.load_vocab()
        self.sentenizer = settings.sentenziser
        self.sentences = self.sentenize()

    def sentenize(self) -> list[Sentence]:
        sentences = []
        for sent in self.sentenizer(self.text):
            sentences.append(Sentence(sent=sent, doc=self))

        return sentences

    def load_vocab(self):

        vocab = []

        if self.settings.tokenize_compounds:
            vocab += load_json(
                filename="compounds.json",
                folder="chem_nlp/data/compounds",
                keys_to_keep=["name"],
                only_values=True,
                ignore={"name": ["mg", "g"]},
                ignore_contains={"name": ["^[A-Za-z]+\(.*"]},
            )

            vocab += load_json(
                filename="synonyms.json",
                folder="chem_nlp/data/compounds",
                keys_to_keep=["synonym"],
                only_values=True,
                ignore_contains={"synonym": ["^[A-Za-z]+\(.*"]},
            )

        if self.settings.tokenize_foods:
            raise Exception("Not implemented yet")

        return list(set(vocab))


class Sentence:

    def __init__(self, doc: ChemDoc, sent: str):
        self.doc = doc
        self.sent = sent
        self.initial_tokens = doc.settings.initial_tokenizer(sent)
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
        tokens = []
        for token in tokenize.by_phrase_match(
            tokens=(self.initial_tokens),
            vocab=self.doc.vocab,
            max_match=self.doc.settings.targets_per_sentence,
            max_sequence=self.doc.settings.max_sequence,
        ):
            tokens.append(Token(sentence=self, char=token))

        return tokens
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
