import spacy
from spacy.training import Example
import random
from models.ingredients.parse_data import load_train_data, parse_training_data


def load_model(model_name="en_core_web_sm"):
    """Load the existing spaCy model"""
    nlp = spacy.load(model_name)
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")
    ner.add_label("FOOD")
    return nlp


def train_ner(nlp, train_data, iterations=30):
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*unaffected_pipes):
        optimizer = nlp.resume_training()
        for iteration in range(iterations):
            random.shuffle(train_data)
            losses = {}
            for text, annotations in train_data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
            print(f"Losses at iteration {iteration}: {losses}")


def save_model(nlp, output_dir):
    nlp.to_disk(output_dir)


def load_and_test_model(model_dir, test_text):
    nlp = spacy.load(model_dir)
    doc = nlp(test_text)
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])


train_data = parse_training_data(load_train_data())
nlp = load_model()
train_ner(nlp, train_data, iterations=10)
save_model(nlp, "models/ingredients/saved_model")
