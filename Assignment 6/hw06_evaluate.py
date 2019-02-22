"""Assignment 06: Spelling Correction.

ECE 590: Natural Language Processing
Patrick Wang
"""

from hw06_solution import suggest


def test_suggestion():
    """Test spelling correction function."""
    words = [
        'hellox',
        'adfnog',
        'abot',
        'garbage'
    ]
    with open('words.txt') as file_obj:
        dictionary = file_obj.read().splitlines()
    for word in words:
        suggestions = suggest(word, dictionary)
        print(word + ':', suggestions)


if __name__ == "__main__":
    test_suggestion()
