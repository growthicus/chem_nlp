import random
import spacy
from spacy.training import Example
from models.ingredients.parse_data import (
    parse_training_data,
    load_validation_data,
    load_eval_data,
)
import logging


logging.basicConfig(level=logging.DEBUG)


def evaluate_ner(nlp, test_data):
    """Evaluates the NER model using provided test data"""
    scorer = spacy.scorer.Scorer(
        nlp
    )  # Ensure scorer is aware of the spaCy model being evaluated
    examples = []  # Initialize a list to store Example objects
    for input_, annot in test_data:
        doc_gold_text = nlp.make_doc(input_)
        gold = Example.from_dict(doc_gold_text, annot)
        pred_value = nlp(input_)
        example = Example(pred_value, gold.reference)
        examples.append(example)

    # Use the scorer to evaluate the examples
    scores = scorer.score(examples)  # This is the correct usage to get the scores
    return scores


def ents_score():
    # Load pre-trained model
    nlp = spacy.load("models/ingredients/saved_model")

    # Assume these functions are available from your module
    validation_data = parse_training_data(load_validation_data())
    random.shuffle(validation_data)

    # Evaluate the model on the validation set
    evaluation_results = evaluate_ner(nlp, validation_data)
    return evaluation_results["ents_per_type"]


def eval_random():
    # Load pre-trained model
    nlp = spacy.load("models/ingredients/saved_model")
    eval_data = load_eval_data()
    random.shuffle(eval_data)

    for ingredients in eval_data[:10]:
        doc = nlp(ingredients["ingredients"].strip())
        logging.debug("###########")
        logging.debug(f"INGREDIENTS: {ingredients['ingredients']}")
        for ent in doc.ents:
            logging.debug(f"{ent.text} - {ent.label_}")


if __name__ == "__main__":
    logging.debug(ents_score())
    logging.debug(eval_random())
