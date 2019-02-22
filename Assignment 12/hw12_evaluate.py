"""Assignment 12: Lexicon Expansion.

ECE 590: Natural Language Processing
Patrick Wang
"""

import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import movie_reviews
from hw12_solution import score_document


def test_lexicon_expansion():
    data = [
        (list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)
    ]
    np.random.shuffle(data)
    documents = [datum[0] for datum in data]
    labels = [datum[1] for datum in data]
    positive_seeds = ['terrific']
    negative_seeds = ['horrible']
    scores = np.array([score_document(doc, positive_seeds, negative_seeds) for doc in documents])

    labels = np.array([0 if label == 'neg' else 1 for label in labels])
    uscores = sorted(list(set(scores)))
    pd = []
    pf = []
    for uscore in uscores:
        pd.append(1 - np.sum(np.logical_and(scores > uscore, labels == 1)) / np.sum(labels))
        pf.append(1 - np.sum(np.logical_and(scores > uscore, labels == 0)) / np.sum(labels))
    plt.plot(pd, pf, '-o')
    plt.xlabel('PD')
    plt.ylabel('PF')
    plt.show()
    top = documents[np.argsort(scores)[-1]]
    print(' '.join(top))

if __name__ == "__main__":
    test_lexicon_expansion()
