from chem_nlp.data.loaders import load_json


def test_load_json():
    _file = "compounds.json"
    folder = "chem_nlp/data/compounds"

    data = load_json(
        filename=_file,
        folder=folder,
        keys_to_keep=["name"],
        only_values=True,
        lower=True,
    )

    assert data[-1] == "tg(a-17:0/a-21:0/a-25:0)[rac]"
