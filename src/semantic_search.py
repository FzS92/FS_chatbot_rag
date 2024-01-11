"""
Semantic Search

This script performs semantic search by measuring the similarity between two texts.
It is using semantic_search method to perform semantic search!

References: My GitHub repository:
https://github.com/FzS92/SemanticSearch/
"""

import logging
import re
from typing import List

import torch
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity

from .model_tokenizer import text_summarization


def remove_extra_newlines(string: str) -> str:
    """
    Removes extra newlines from a string.

    Args:
        string (str): The input string.

    Returns:
        str: The string with extra newlines removed.
    """
    return "\n".join(line for line in string.splitlines() if line.strip())


def separate_paragraphs(text: str) -> List[str]:
    """
    Separates a text into unique paragraphs.

    Args:
        text (str): The input text.

    Returns:
        List[str]: A list of unique paragraphs from the input text.
    """
    paragraphs = text.split("\n")
    paragraphs = list(set(paragraphs))
    return [s for s in paragraphs if s != "" or s != ""]


def filter_strings_by_word_count(strings: List[str]) -> List[str]:
    """
    Filter a list of strings to remove elements with 22 words or less.
    It's a function to remove short paragraphs.

    Parameters:
    - strings (List[str]): The input list of strings.

    Returns:
    - List[str]: The filtered list of strings.
    """
    min_number_of_words = 22
    filtered_strings = [s for s in strings if len(s.split()) > min_number_of_words]
    return filtered_strings


def get_unique_sentences(text: str) -> List[str]:
    """
    Split the input text into sentences, remove leading empty lines,
    and return a list of unique sentences.

    Args:
        text (str): The input text containing sentences.

    Returns:
        List[str]: A list of unique sentences.

    """
    text = text.lstrip("\n")  # Remove leading empty lines
    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    unique_sentences = list(set(sentences))
    return unique_sentences


def encode_text(text: List[str], model: SentenceTransformer) -> torch.Tensor:
    """
    Encode the given list of texts using the specified model.

    Args:
        text (List[str]): The input texts to be encoded.
        model (SentenceTransformer): The model used for encoding.

    Returns:
        torch.Tensor: The encoded texts as a tensor.
    """
    with torch.no_grad():
        outputs = model.encode(text)
        outputs = torch.tensor(outputs)
    return outputs


def get_top_n_cosine_similarity_rows(
    tensor1: torch.Tensor, tensor2: torch.Tensor, topk: int
) -> torch.Tensor:
    """
    Get the indices of the top N rows with highest cosine similarity scores
    between two tensors.

    Args:
        tensor1 (torch.Tensor): The first tensor.
        tensor2 (torch.Tensor): The second tensor.
        n (int): The number of top rows to retrieve.

    Returns:
        torch.Tensor: The indices of the top N rows.
    """
    similarity_scores = cosine_similarity(tensor1, tensor2)
    top_n_indices = torch.topk(similarity_scores.squeeze(), topk)[1]
    return top_n_indices


def semantic_search(
    model_name: str, mode: str, searching_for: str, text: str, n_similar_texts: int = 5
) -> str:
    """
    Perform semantic search by measuring the similarity between the
    searching_for text and the text corpus.

    Args:
        model_name (str): The name of the SentenceTransformer model.
                          "all-mpnet-base-v2" is recommended.
        mode (str): Search for most similar sentences or paragraphs?
        searching_for (str): The text to search for (query).
        text (str): The corpus of text to search in.
        n_similar_texts (int) = number of top search results

    Returns:
        str: The search results as a formatted string.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name).to(device)

    text = remove_extra_newlines(text)
    if mode == "Paragraph":
        text = separate_paragraphs(text)
        text = filter_strings_by_word_count(text)
    elif mode == "Sentence":
        text = get_unique_sentences(text)

    search_and_text = [searching_for]
    search_and_text.extend(text)

    encoded = encode_text(search_and_text, model)

    encoded_search = encoded[0]  # Encode of searching_for
    encoded_text = encoded[1:]  # Encode of text

    n_similar_texts = min(n_similar_texts, len(text))

    index_of_similar = get_top_n_cosine_similarity_rows(
        encoded_search, encoded_text, n_similar_texts
    )

    # output = f"Top {n_similar_texts} are as follows:\n\n"
    output = ""
    old_output = ""
    for i in range(n_similar_texts):
        # output += f"Match number {i+1}:\n"
        try:
            old_output += text[index_of_similar[i]] + "\n\n"
            output += summarize_text(text[index_of_similar[i]]) + "\n\n"
        except:
            pass

    # Configuring logging
    logging.basicConfig(level=logging.INFO)

    logging.info("=" * 10)
    logging.info("Top paragraphs search results before summarization:\n" + old_output)
    logging.info("=" * 10)
    logging.info("Top paragraphs search results after summarization:\n" + output)
    logging.info("=" * 10)

    return output


def summarize_text(
    text: str, max_length: int = 230, min_length: int = 15, do_sample: bool = False
) -> str:
    """
    Summarizes the input text using the Hugging Face summarization pipeline.

    Args:
        text (str): The input text to be summarized.
        max_length (int, optional): The maximum length of the summary. Defaults to 230.
        min_length (int, optional): The minimum length of the summary. Defaults to 15.
        do_sample (bool, optional): If True, uses sampling to generate the summary. Defaults to False.

    Returns:
        str: The summarized text.
    """
    model, _ = text_summarization()
    with torch.no_grad():
        summary = model(
            text, max_length=max_length, min_length=min_length, do_sample=do_sample
        )
    return summary[0]["summary_text"]
