"""Assignment 01: Hash Functions.

ECE 590: Natural Language Processing
Patrick Wang
"""

import random
import numpy as np

from hw01_solution import to_str_invertible, to_int_invertible, to_int_uniform, to_int_nonuniform


NUM_CODE_POINTS = int('0x110000', 16)


def random_unicode(num_chars):
    """Generate a random Unicode character."""
    return ''.join([chr(random.randrange(NUM_CODE_POINTS)) for _ in range(num_chars)])


def test_uniform():
    """Test to_int_uniform() function."""
    # generate random unicode strings
    strings = [random_unicode(4) for _ in range(1000)]
    # has strings to ints
    values = [to_int_uniform(string) for string in strings]
    print('\nShould be close to 500:')
    print(sum(values))


def test_nonuniform():
    """Test to_int_nonuniform() function."""
    # generate random unicode strings
    strings = [random_unicode(4) for _ in range(1000)]
    # hash strings to integers
    values = [to_int_nonuniform(string) for string in strings]
    # compute histogram of integers
    hist, _ = np.histogram(values, bins=100, range=(0, 100))
    print('\nShould be nonuniform:')
    print(hist)


def test_invertible():
    """Test to_int_invertible() and to_str_invertible() functions."""
    # hash string to integer
    value = to_int_invertible('hello')
    # un-hash integer back to string
    string = to_str_invertible(value)
    print('\nShould say "hello":')
    print(string)


if __name__ == "__main__":
    test_uniform()
    test_nonuniform()
    test_invertible()
