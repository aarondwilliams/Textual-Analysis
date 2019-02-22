"""Assignment 07: Spelling Correction.

ECE 590: Natural Language Processing
Patrick Wang
"""

import re
import nltk
import numpy as np
from nltk import word_tokenize
from hw07_solution import transition_matrix, most_likely_word


def test_tmat():
    """Test transition matrix."""
    corpus = nltk.corpus.gutenberg.raw('austen-sense.txt')
    corpus = word_tokenize(corpus.lower())
    corpus = [word for word in corpus if re.fullmatch('[a-z]+', word)]
    transitions = transition_matrix(corpus)

    try:
        import matplotlib.pyplot as plt
        plt.imshow(transitions)
        plt.xlabel('to')
        plt.ylabel('from')
        plt.title('transition probability')
        plt.xticks(ticks=range(27), labels=[chr(i) for i in range(97, 123)])
        plt.yticks(ticks=range(27), labels=[chr(i) for i in range(97, 123)])
        plt.show()
    except:
        print("matplotlib may not be correctly set up.")

    return transitions


def test_ml(transitions):
    with open('words.txt') as file_obj:
        dictionary = file_obj.read().splitlines()
    goals = [
        't',
        'he',
        'the',
        'here',
        'there',
        'washer',
        'heather',
        'wherever',
        'therefore',
        'manchester'
    ]
    for n in range(1, 11):
        word = most_likely_word(dictionary, transitions, n=n)
        success = "✔" if word == goals[n - 1] else "✘"
        print(f"{n:02d}: {word} {success}")


if __name__ == "__main__":
    transitions = test_tmat()
    test_ml(transitions)
