from models.ingredients.parse_data import (
    load_train_data,
    parse_training_data,
)


def test_load_train():
    assert load_train_data()


def test_create_training_data():
    data = [
        {
            "ingredients": "apple, banana, also contains (apple)",
            "labels": "apple, banana, apple",
        }
    ]
    parsed = parse_training_data(data)
    assert parsed[0][0] == "apple, banana, also contains (apple)"
    assert len(parsed[0][1]["entities"]) == 3
    for spans in parsed[0][1]["entities"]:
        assert parsed[0][0][spans[0] : spans[1]] in ["apple", "banana"]
