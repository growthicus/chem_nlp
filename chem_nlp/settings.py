from chem_nlp.tokenizers import entity, tokenize, vocab
from chem_nlp.sentenzisers import sentenzise
from chem_nlp.core import Settings


settings = Settings(
    sentenziser=sentenzise.standard,
    ignore_case=True,
    token_split_patterns=[tokenize.TM_UNIT],
    token_merge_vocabs=[vocab.V_COMPOUND, vocab.V_COMPOUND_SYNONYM, vocab.V_FOOD],
    entity_patterns=[
        entity.EM_COMPOUND,
        entity.EM_COMPOUND_SYN,
        entity.EM_FOOD,
        entity.EM_MODIFIER,
        entity.EM_QUALIFIER,
    ],
    targets_per_sentence=2,
    min_vocab_word_length=4,
)

"""
settings = Settings(
    ignore_case=True,
    token_split_patterns=[tokenize.TM_UNIT],
    token_merge_vocabs=[vocab.V_COMPOUND, vocab.V_COMPOUND_SYNONYM, vocab.V_FOOD],
    entity_patterns=[
        entity.EM_COMPOUND,
        entity.EM_COMPOUND_SYN,
        entity.EM_FOOD,
        entity.EM_MODIFIER,
        entity.EM_QUALIFIER,
    ],
    targets_per_sentence=2,
    min_vocab_word_length=4,
)
"""
