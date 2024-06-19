from __future__ import annotations
from chem_nlp.tokenizers import tokenize, entity
from chem_nlp.tokenizers import vocab
from chem_nlp.dc import Settings
import nltk
import itertools
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
        self.vocab = self.load_vocabs()
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

    def load_vocabs(self):

        all_words = []
        for vocab in self.settings.token_merge_vocabs:
            if self.settings.ignore_case:
                vocab.to_lower = True

            data = vocab.load()
            all_words.extend(data)

        return list(set(all_words))


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
            vocab=(self.doc.vocab),
            char_tokens=(self.initial_tokens),
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

        entity.by_word_pattern(
            tokens=self.tokens,
            patterns=self.doc.settings.entity_patterns,
            settings=self.doc.settings,
            max_match=self.doc.settings.targets_per_sentence,
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
