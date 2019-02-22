"""Assignment 09: Word Features.

ECE 590: Natural Language Processing
Patrick Wang
"""

import nltk
from hw09_solution import related_words


def test_related():
    word = 'play'
    print(f'related to {word}: {related_words(word)}')
    word = 'jury'
    print(f'related to {word}: {related_words(word)}')


if __name__ == "__main__":
    test_related()
