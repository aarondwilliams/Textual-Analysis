"""Assignment 02: Regular Expressions.

ECE 590: Natural Language Processing
Patrick Wang
"""

import re


with open('example.py') as text_file:
    TEXT = text_file.readlines()


def test1():
    """Test first pattern."""
    with open('solution1.txt') as file:
        pattern1 = file.readline()

    print('\nChecking variable match...')
    truth = [['a'], ['a'], ['b'], ['a'], ['c', 'a', 'b'], ['d', 'x'], ['x'], ['a2', 'B_2'], ['g', 'h', 'i']]
    fail = False
    for idx, line in enumerate(TEXT):
        match = re.findall(pattern1, line)
        truth_string = " and ".join([f"'{x}'" for x in truth[idx]]) or 'nothing'
        guess_string = " and ".join([f"'{x}'" for x in match]) or 'nothing'
        if match != truth[idx]:
            fail = True
            print(f"{truth_string} in line {idx + 1} should match, but you matched {guess_string}.")
    if not fail:
        print('Success!')


def test2():
    """Test second pattern."""
    with open('solution2.txt') as file:
        pattern2 = file.readline()

    print('\nChecking assignment line match...')
    truth = [True, False, True, False, True, True, False, True, True]
    fail = False
    for idx, line in enumerate(TEXT):
        match = re.findall(pattern2, line)
        if (len(match) > 0) != truth[idx]:
            fail = True
            print(f"Line {idx + 1} should{'' if truth[idx] else ' not'} be matched but your pattern did{'' if match is not None else ' not'}.")
    if not fail:
        print('Success!')


def test3():
    """Test third pattern."""
    with open('solution3.txt') as file:
        pattern3 = file.readline()

    print('\nChecking assignment variable match...')
    truth = [['a'], [], ['b'], [], ['c'], ['d'], [], ['a2', 'B_2'], ['g', 'h', 'i']]
    fail = False
    for idx, line in enumerate(TEXT):
        match = re.findall(pattern3, line)
        truth_string = " and ".join([f"'{x}'" for x in truth[idx]]) or 'nothing'
        guess_string = " and ".join([f"'{x}'" for x in match]) or 'nothing'
        if match != truth[idx]:
            fail = True
            print(f"{truth_string} in line {idx + 1} should match, but you matched {guess_string}.")
    if not fail:
        print('Success!')


if __name__ == "__main__":
    test1()
    test2()
    test3()
