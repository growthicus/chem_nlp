import Levenshtein as lev
import logging
from nltk.util import trigrams

logging.basicConfig(level=logging.DEBUG)


def filter_unique_n_gram(matches: list[dict]):
    # Dictionary to store the best (lowest lvd) entries for each unique n_gram
    n_gram_dict = {}

    for item in matches:
        n_gram_key = item[
            "n_gram"
        ].lower()  # Normalize the n_gram for case-insensitive comparison
        # Check if we already have this n_gram and if the current lvd is lower than the stored one
        if (
            n_gram_key not in n_gram_dict
            or item["lvd"] < n_gram_dict[n_gram_key]["lvd"]
        ):
            n_gram_dict[n_gram_key] = item

    # Return the filtered list of dictionaries
    return list(n_gram_dict.values())


def find_tokens_by_trigram(tokens: list[str], vocab: list[str], lvt=1) -> list[str]:
    matches = []
    for i, trigram in enumerate(list(trigrams(tokens))):
        # logging.info(f"TRIGRAM: {trigram}")
        trigram_matches = []
        for i in range(len(trigram)):
            for j in range(i + 1, len(trigram) + 1):
                n_gram = " ".join(trigram[i:j])
                for word in vocab:
                    lvd = lev.distance(n_gram.lower(), word)
                    if lvd <= lvt:
                        trigram_matches.append(
                            {"n_gram": n_gram, "lvd": lvd, "new": word}
                        )

        if trigram_matches:
            matches.append(sorted(trigram_matches, key=lambda x: x["lvd"])[0])

    return filter_unique_n_gram(matches)


def by_phrase_match(tokens: list[str], vocab: list[str]) -> list[str]:
    matches = find_tokens_by_trigram(tokens=tokens, vocab=vocab)
    sorted_matches = sorted(matches, key=lambda x: x["lvd"])
    logging.error(sorted_matches)

    for merge_dict in sorted_matches:
        n_gram = merge_dict["n_gram"]
        new_value = merge_dict["new"]
        words_to_merge = n_gram.split()
        length = len(words_to_merge)

        # Find the starting tokens of the sequence in the list
        for i in range(len(tokens) - length + 1):
            # Check if the subsequent elements in the list match the words to merge
            if tokens[i : i + length] == words_to_merge:
                # Replace the specific range with the new_value
                tokens[i : i + length] = [new_value]
                # Continue to next merge_dict without breaking to allow further replacements
                continue

    logging.info(tokens)
    return tokens


"""                 90              100                 75
global_matches = [i really enjoy, really enjoy, enjoy taking vitamin]
('I', 'really', 'enjoy')
    i 70
    i really 75
    i really enjoy 90
('really', 'enjoy', 'taking')
    really 60
    really enjoy 100
    really enjoy taking 75
('enjoy', 'taking', 'vitamin')
    enjoy 40
    enjoy taking 30
    enjoy taking vitamin 80
('taking', 'vitamin', 'a')
('vitamin', 'a', 'every')
('a', 'every', 'day')
('every', 'day', '.')
"""
