import pytest

from spam_classifier.text_cleaning import clean_text, proportion_of_numbers, average_word_length


def test_clean_text_basic():
    assert clean_text("Hello, WORLD!!!") == "hello world"


def test_proportion_of_numbers():
    assert proportion_of_numbers("a 1 b 2 c") == 0.4


def test_average_word_length():
    val = average_word_length("ab cdef gh")
    # lengths: 2,4,2 -> mean = 2.666...
    assert 2.5 < val < 2.8


