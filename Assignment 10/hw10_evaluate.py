"""Assignment 10: Latent Dirichlet Allocation.

ECE 590: Natural Language Processing
Patrick Wang
"""

import matplotlib.pyplot as plt
import numpy as np
from hw10_solution import lda


def test_lda():
    vocabulary = ['bass', 'pike', 'deep', 'tuba', 'horn', 'catapult']
    V = len(vocabulary)
    phi = np.array([
        [0.4, 0.4, 0.2, 0.0, 0.0, 0.0],
        [0.0, 0.3, 0.1, 0.0, 0.3, 0.3],
        [0.3, 0.0, 0.2, 0.3, 0.2, 0.0]
    ])
    theta = np.array([
        [0.0, 1.0, 0.0],
        [0.7, 0.3, 0.0],
        [0.0, 0.2, 0.8]
    ])
    N = 10000
    documents = lda(vocabulary, phi, theta, N)
    fig, ax = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True, sharey=True,
        figsize=(9, 7)
    )
    for i, doc in enumerate(documents):
        wi = [vocabulary.index(word) for word in doc]
        n, bins, _ = ax[i].hist(
            wi, np.linspace(-0.5, V - 0.5, V + 1),
            rwidth=0.8)
        ax[i].set_xticks(list(range(V)))
        ax[i].set_xticklabels(vocabulary)
        ax[i].set_title(f"document {i}:")
    plt.show()


if __name__ == "__main__":
    test_lda()
