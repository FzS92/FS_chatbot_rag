"""
This file contains unit tests for the functions in the 'semantic_search' module.
Other methods have been tested before.
"""

from typing import List

import pytest

from .semantic_search import filter_strings_by_word_count


def test_filter_strings_by_word_count():
    input_strings_1 = ["Short string"]

    expected_result = []
    assert filter_strings_by_word_count(input_strings_1) == expected_result

    input_strings_2 = ["Long string with more than 22 words" * 5]
    assert filter_strings_by_word_count(input_strings_2) == input_strings_2
