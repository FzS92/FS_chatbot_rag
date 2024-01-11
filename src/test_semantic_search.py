"""
This file contains tests for some functions in the 'semantic_search' module.
"""

from typing import List

from sentence_transformers import SentenceTransformer

from .semantic_search import (
    encode_text,
    filter_strings_by_word_count,
    remove_extra_newlines,
    separate_paragraphs,
)


def test_remove_extra_newlines():
    input_string = "Hello\n\n\nWorld\n\n\n"
    expected_output = "Hello\nWorld"
    assert remove_extra_newlines(input_string) == expected_output


def test_separate_paragraphs():
    input_text = "Hello\n\nWorld\n\nHello\n\nWorld"
    expected_output = ["Hello", "World"]
    assert set(separate_paragraphs(input_text)) == set(expected_output)


def test_filter_strings_by_word_count():
    input_strings = ["Short string", "Long string with more than 22 words" * 4]
    expected_output = ["Long string with more than 22 words" * 4]
    assert filter_strings_by_word_count(input_strings) == expected_output


def test_encode_text():
    model = SentenceTransformer("all-mpnet-base-v2")
    input_text = ["Hello", "World"]
    encoded_text = encode_text(input_text, model)
    assert len(encoded_text) == len(input_text)
