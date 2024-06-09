import Levenshtein as lev
import logging
from nltk.util import bigrams, trigrams

logging.basicConfig(level=logging.DEBUG)


def find_tokens_by_sequence(
    tokens: list[str], vocab: list[str], max_sequence: int, max_match: int
) -> list[str]:

    if max_sequence == 2:
        gramify = bigrams
    elif max_sequence == 3:
        gramify = trigrams

    matches = []
    for i, x_gram in enumerate(list(gramify(tokens))):
        for i in range(len(x_gram)):
            for j in range(i + 1, len(x_gram) + 1):
                n_gram = " ".join(x_gram[i:j]).lower()
                if n_gram in vocab:
                    matches.append(n_gram)
                    if len(matches) == max_match and len(x_gram) == max_sequence:
                        return matches

    return matches


def by_phrase_match(
    tokens: list[str], vocab: list[str], max_sequence: int = 3, max_match: int = 1
) -> list[str]:
    l_tokens = [t.lower() for t in tokens]
    matches = find_tokens_by_sequence(
        tokens=l_tokens, vocab=vocab, max_sequence=max_sequence, max_match=max_match
    )

    for match in sorted(matches, key=len):
        words_to_merge = match.split()
        length = len(words_to_merge)

        # Find the starting tokens of the sequence in the list
        for i in range(len(l_tokens) - length + 1):
            logging
            # Check if the subsequent elements in the list match the words to merge
            if l_tokens[i : i + length] == words_to_merge:
                # Replace the specific range with the new_value
                l_tokens[i : i + length] = [match]
                # Continue to next merge_dict without breaking to allow further replacements
                continue

    logging.debug(l_tokens)
    return l_tokens


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
