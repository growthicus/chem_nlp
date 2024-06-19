from chem_nlp.dc import Vocab
import logging


logging.basicConfig(level=logging.DEBUG)

V_COMPOUND = Vocab(
    name="compound",
    filename="compound.csv",
    folder="chem_nlp/data/compound",
    keys_to_keep=["name"],
    only_values=True,
    ignore_contains={"name": [r"^[A-Za-z]+\(.*"]},
)

V_COMPOUND_SYNONYM = Vocab(
    name="compound_synonym",
    filename="compound_syn.csv",
    folder="chem_nlp/data/compound",
    keys_to_keep=["name"],
    only_values=True,
    ignore={"name": ["Mg", "G"]},
    ignore_contains={"name": [r"^[A-Za-z]+\(.*"]},
)


V_FOOD = Vocab(
    name="food",
    filename="food.csv",
    folder="chem_nlp/data/food",
    keys_to_keep=["name"],
    only_values=True,
    split_to_nouns=True,
)

V_UNIT_WEIGHT = Vocab(
    name="weight",
    filename="weight.json",
    folder="chem_nlp/data/units",
    keys_to_keep=["name"],
    only_values=True,
)

V_MODIFIER = Vocab(
    name="modifier",
    filename="modifier.json",
    folder="chem_nlp/data/misc",
    keys_to_keep=["name"],
    only_values=True,
)

V_NM_POSTFIX = Vocab(
    name="nomenclature_postfix",
    filename="nomenclature_postfix.json",
    folder="chem_nlp/data/misc",
    keys_to_keep=["name"],
    only_values=True,
)
