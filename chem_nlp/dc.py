from __future__ import annotations

from typing import Callable, Union
from dataclasses import dataclass, field
from chem_nlp.sentenzisers import sentenzise
from chem_nlp.data.loaders import load_csv, load_json
import itertools
import nltk


@dataclass
class Vocab:
    name: str
    filename: str
    folder: str
    to_lower: bool = False
    keys_to_keep: Union[list, None] = None
    only_values: bool = True
    ignore: Union[dict, None] = None
    ignore_contains: Union[dict, None] = None
    split_to_nouns: bool = False
    data: set = field(default_factory=set)
    data_nouns: list = field(default_factory=list)

    def load(self):
        loader = load_csv if "csv" in self.filename else load_json
        if not self.data:
            self.data = loader(
                filename=self.filename,
                folder=self.folder,
                to_lower=self.to_lower,
                keys_to_keep=self.keys_to_keep,
                only_values=self.only_values,
                ignore=self.ignore,
                ignore_contains=self.ignore_contains,
            )

        return self.data

    def nouns(self):

        if not self.data:
            raise ValueError("No data loaded")

        if not self.data_nouns and self.split_to_nouns:
            words = itertools.chain(*[word.split() for word in self.data])
            pos = nltk.pos_tag(list(words))
            self.data_nouns = [pos[0] for pos in pos if pos[1] == "NN"]

        return self.data_nouns


@dataclass
class Settings:
    sentenziser: Callable = sentenzise.standard
    initial_tokenizer: Callable = nltk.word_tokenize
    ignore_case: bool = False

    token_merge_vocabs: list[Vocab] = field(default_factory=list)
    token_split_patterns: list[Vocab] = field(default_factory=list)
    entity_patterns: list[Matcher] = field(default_factory=list)

    targets_per_sentence: int = 1
    max_sequence_length: int = 3
    min_vocab_word_length: int = 3

    def __post_init__(self):

        for matcher in self.entity_patterns:
            matcher.settings = self


@dataclass
class Matcher:
    name: str
    pattern_funcs: list[Callable]
