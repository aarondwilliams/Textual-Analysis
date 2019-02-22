import nltk
from nltk import word_tokenize

from hw04_solution import finish_sentence


def test_generator():
    """Test Markov text generator."""
    corpus = nltk.corpus.gutenberg.raw('austen-sense.txt')
    corpus = word_tokenize(corpus.lower())

    words = finish_sentence(['she', 'was', 'not'], 3, corpus)
    print(' '.join(words))
    if len(words) == 39:
        print('Success!')
    else:
        print('The result is expected to have 39 "words".')


if __name__ == "__main__":
    test_generator()
