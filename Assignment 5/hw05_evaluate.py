"""Assignment 05: Edit Distance.

ECE 590: Natural Language Processing
Patrick Wang
"""

from hw05_solution import levenshtein_dp


def levenshtein(a, b):
    """Compute levenshtein distance using recursion."""
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    cost = 0 if a[-1] == b[-1] else 1
    return min((
        levenshtein(a[:-1], b) + 1,
        levenshtein(a, b[:-1]) + 1,
        levenshtein(a[:-1], b[:-1]) + cost
    ))


def test_levenshteins():
    """Test Levenshtein edit distance functions."""
    pairs = [
        ['hello', 'cello'],
        ['hello', 'shallow'],
        ['hello', 'he']
    ]
    for pair in pairs:
        a = pair[0]
        b = pair[1]
        d = levenshtein(a, b)
        d2 = levenshtein_dp(a, b)
        print(f"The distance between '{a}' and '{b}' is {d}.")
        print(f"  Your function gave: {d2}")


if __name__ == "__main__":
    test_levenshteins()
