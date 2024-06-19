import itertools


def new_lines(text):
    return text.split("\n")


def comma(text):
    return text.split(",")


def standard(text):
    # splitting text on comma, punkt and new line
    return list(itertools.chain(*[comma(sent) for sent in new_lines(text)]))
