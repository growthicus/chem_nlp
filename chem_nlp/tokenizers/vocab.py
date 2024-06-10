from chem_nlp.dc import Vocab


V_COMPOUND = Vocab(
    name="compound",
    entity="COMPOUND",
    filename="compound.json",
    folder="chem_nlp/data/compounds",
    keys_to_keep=["name"],
    only_values=True,
    ignore_contains={"name": [r"^[A-Za-z]+\(.*"]},
)

V_COMPOUND_SYNONYM = Vocab(
    name="compound_synonym",
    entity="COMPOUND_SYN",
    filename="synonym.json",
    folder="chem_nlp/data/compounds",
    keys_to_keep=["synonym"],
    only_values=True,
    ignore={"synonym": ["Mg", "G"]},
    ignore_contains={"synonym": [r"^[A-Za-z]+\(.*"]},
)


V_FOOD = Vocab(
    name="food",
    entity="FOOD",
    filename="food.json",
    folder="chem_nlp/data/food",
    keys_to_keep=["name", "name_scientific"],
    only_values=True,
)

V_UNIT_WEIGHT = Vocab(
    name="weight",
    entity="weight",
    filename="weight.json",
    folder="chem_nlp/data/units",
    keys_to_keep=["name"],
    only_values=True,
)
