"""Assignment 11: Latent Dirichlet Allocation #2.

ECE 590: Natural Language Processing
Patrick Wang
"""

import matplotlib.pyplot as plt
import numpy as np
from hw11_solution import lda, lda_solve


def test_lda_solve():
    vocabulary = ['bass', 'pike', 'deep', 'tuba', 'horn', 'catapult']
    V = len(vocabulary)
    phi = np.array([
        [0.4, 0.4, 0.2, 0.0, 0.0, 0.0],
        [0.0, 0.3, 0.1, 0.0, 0.3, 0.3],
        [0.3, 0.0, 0.2, 0.3, 0.2, 0.0]
    ])
    alpha = [1, 2, 3]
    num_docs = 1000
    theta = []
    for _ in range(num_docs):
        theta.append(np.random.dirichlet(alpha))
    theta = np.array(theta)
    N = 100
    documents = lda(vocabulary, phi, theta, N)
    phi_est, words = lda_solve(documents)

    # sort
    order = np.argsort(words)
    phi_est = phi_est[:, order]
    words = [words[i] for i in order]

    # display topics
    plt.imshow(phi_est / np.sum(phi_est, axis=1, keepdims=True))
    plt.xticks(ticks=range(6), labels=words)
    plt.yticks(ticks=range(3), labels=[f"topic {i}" for i in range(3)])
    plt.show()


if __name__ == "__main__":
    test_lda_solve()
