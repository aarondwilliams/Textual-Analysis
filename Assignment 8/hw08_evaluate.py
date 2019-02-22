"""Assignment 08: Viterbi.

ECE 590: Natural Language Processing
Patrick Wang
"""

import numpy as np
from hw08_solution import infer_states


def draw_from_categorical(pmf):
    """Draw from a categorical distribution with given pmf."""
    d = len(pmf)
    x = np.random.uniform()
    cmf = np.cumsum(pmf)
    for i, p in enumerate(cmf):
        if x < p:
            return i


def draw_state_sequence(A, pi, n=10):
    """Draw a state sequence of length n with state transition probability matrix A and starting state distribution pi."""
    states = [draw_from_categorical(pi)]
    for _ in range(n - 1):
        states.append(draw_from_categorical(A[states[-1]]))
    return states


def draw_obs(states, B):
    """Draw observations for given states with observation probability matrix B."""
    return [draw_from_categorical(B[i]) for i in states]


def test_simple():
    """Test the simple HMM example from Wikipedia.

    https://en.wikipedia.org/wiki/Viterbi_algorithm#Example
    """
    # pi_i = probability of starting at state i
    pi = [0.6, 0.4]
    # a_{ij} = probability of transitioning from state i to state j
    A = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])
    # b_{ik} = probability of observing k at state i
    B = np.array([
        [0.1, 0.4, 0.5],
        [0.6, 0.3, 0.1]
    ])

    np.random.seed(2018)
    # states = draw_state_sequence(A, pi)
    # obs = draw_obs(states, B)
    obs = [2, 1, 0]
    states_guess, prob = infer_states(obs, pi, A, B)
    state_map = {0: 'healthy', 1: 'fever'}
    print(f'{[state_map[s] for s in states_guess]} with p = {prob}')
    if states_guess == [0, 0, 1] and round(prob, 5) == 0.01512:
        print('PASS')
    else:
        print('FAIL')


if __name__ == "__main__":
    test_simple()
