from typing import Callable, Union
from dataclasses import dataclass, field
from chem_nlp.sentenzisers import sentenzise
from chem_nlp.data.loaders import load_json

import nltk


@dataclass
class Vocab:
    name: str
    entity: str
    filename: str
    folder: str
    to_lower: bool = False
    keys_to_keep: Union[list, None] = None
    only_values: bool = True
    ignore: Union[dict, None] = None
    ignore_contains: Union[dict, None] = None
    data: list = field(default_factory=list)

    def load(self):
        if not self.data:
            self.data = load_json(
                filename=self.filename,
                folder=self.folder,
                to_lower=self.to_lower,
                keys_to_keep=self.keys_to_keep,
                only_values=self.only_values,
                ignore=self.ignore,
                ignore_contains=self.ignore_contains,
            )

        return self.data


@dataclass
class EntityMatcher:
    name: str
    entity: str
    pattern_funcs: list[Callable]


@dataclass
class Settings:
    sentenziser: Callable = sentenzise.new_lines
    initial_tokenizer: Callable = nltk.word_tokenize
    ignore_case: bool = False
    token_vocabs: list[Vocab] = field(default_factory=list)
    entity_patterns: list[EntityMatcher] = field(default_factory=list)
    targets_per_sentence: int = 1
    max_sequence_length: int = 3
    min_vocab_word_length: int = 3
