import re
import nltk

from hw03_solution import modernize

def test_modernizer():
    """Test Shakespearean spelling modernizer."""
    macbeth = nltk.corpus.gutenberg.raw('shakespeare-macbeth.txt')
    scene5 = macbeth[macbeth.find('Scena Quinta.') + len('Scena Quinta.'):macbeth.find('Scena Sexta.')]

    # attempt
    attempt = modernize(scene5)

    # get target
    with open('target.txt') as file:
        target = file.read()

    # ignore whitespace
    attempt = re.sub(r'\s+', r' ', attempt)
    target = re.sub(r'\s+', r' ', target)

    # ignore punctuation
    attempt = re.sub(r'[.,;:?]+', r'', attempt)
    target = re.sub(r'[.,;:?]+', r'', target)

    print('Success!' if attempt == target else 'Failure')

if __name__ == "__main__":
    test_modernizer()
