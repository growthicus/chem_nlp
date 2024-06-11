from __future__ import annotations
from collections import defaultdict
from chem_nlp.tokenizers import tokenize, entity
from chem_nlp.dc import Settings
import nltk
import logging


logging.basicConfig(level=logging.DEBUG)

# Download necessary NLTK data files
nltk.download("averaged_perceptron_tagger")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")


class ChemDoc:

    def __init__(self, text: str, settings: Settings):
        self.text = text
        self.settings = settings
        self.vocab = self.load_vocab()
        self.sentenizer = settings.sentenziser
        self.sentences = self.sentenize()

    def sentenize(self) -> list[ChemSentence]:
        sentences = []
        for sent in self.sentenizer(self.text):
            sentences.append(
                ChemSentence(
                    sent=sent.lower() if self.settings.ignore_case else sent, doc=self
                )
            )

        return sentences

    def load_vocab(self) -> dict:

        vocabs = defaultdict(list)
        for vocab in self.settings.token_merge_vocabs:
            vocab.to_lower = self.settings.ignore_case
            vocabs[vocab.name] += vocab.load()

        return vocabs

    def vocab_to_list(self) -> list:

        vocabs = []
        for values in self.vocab.values():
            vocabs += values

        return list(set(vocabs))


class ChemSentence:

    def __init__(self, doc: ChemDoc, sent: str):
        self.doc = doc
        self.sent = sent
        self.initial_tokens = doc.settings.initial_tokenizer(sent)
        self.pos_tags = self.set_pos_tags(self.initial_tokens)
        self.tokens = self.tokenize()
        self.entity_recognition()

    def is_first(self):
        return 0 == self.doc.sentences.index(self)

    def get_pos_tag(self, token: ChemToken) -> str:
        # This can return the wrong POS if
        # the word occour more than once
        # and has differnt POS for each
        # occourence

        for word, pos in self.pos_tags:
            if word == token.char:
                if token.entity in ["COMPOUND", "COMPOUND_SYN", "FOOD"]:
                    return "NN"
                return pos

    def set_pos_tags(self, tokens: list[str]):
        # Get the POS tags
        pos_tags = nltk.pos_tag(tokens)
        return pos_tags

    def tokenize(self) -> list[ChemToken]:
        tokens = []

        # Merges tokens into bigger grams
        for merged_char in tokenize.merge_by_word_sqequence(
            char_tokens=(self.initial_tokens),
            vocab=self.doc.vocab_to_list(),
            max_match=self.doc.settings.targets_per_sentence,
            max_sequence=self.doc.settings.max_sequence_length,
            min_n_gram_len=self.doc.settings.min_vocab_word_length,
        ):
            # Split tokens into smaller grams
            for divided_char in tokenize.split_by_char_pattern(
                char_token=merged_char, patterns=self.doc.settings.token_split_patterns
            ):
                tokens.append(ChemToken(doc=self.doc, sentence=self, char=divided_char))

        # Re init POS
        self.pos_tags = self.set_pos_tags([token.char for token in tokens])

        return tokens

    def entity_recognition(self):
        entity.by_token_pattern(
            tokens=self.tokens, entity_patterns=self.doc.settings.entity_patterns
        )


class ChemToken:

    def __init__(self, doc: ChemDoc, sentence: ChemSentence, char: str):
        self.doc = doc
        self.sentence = sentence
        self.char = char
        self.entity: str = None

    @property
    def pos(self):
        return self.sentence.get_pos_tag(self)

    def __repr__(self) -> str:
        return f"ChemToken(char={self.char}, pos={self.pos}, entity={self.entity})"
