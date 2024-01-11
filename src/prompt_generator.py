"""
This script provides a function for obtaining information from online sources related
to a given query.
The `prompt_with_rag` function utilizes online search and semantic search to retrieve
relevant information.
"""

from typing import List

# import torch
from .online_search import get_website_text, google_search
from .semantic_search import semantic_search

# from .model_tokenizer import chat_model_tokenizer


def prompt_with_rag(
    query: str, use_google: bool, search_time: str, num_results: int = 3
) -> str:
    """
    Get information from online sources related to the given query.

    Parameters:
    - query (str): The user's query.
    - num_results (int): The maximum number of online search results to consider.

    Returns:
    - str: A string which is prompt engineering of the user query + information retrieved from online sources.
    """

    all_urls = ""
    answer = ""
    loop_count = 0
    if use_google:
        all_urls = google_search(query, search_time)
        num_results = min(
            num_results, len(all_urls)
        )  # Making sure that we got some results

        while len(answer.split()) < 500 and loop_count < num_results:
            url_in_use = all_urls[loop_count]
            top_text = get_website_text(url_in_use)

            top_answers = semantic_search(
                model_name="all-mpnet-base-v2",
                mode="Paragraph",
                searching_for=query,
                text=top_text,
                n_similar_texts=5,
            )

            # Combine the top answers into the final answer
            answer += (
                f"Information from online source {loop_count+1}: \n\n"
                + top_answers
                + "\n\n"
            )

            loop_count += 1
        prompt = f"User: {query}\n\n" + answer + "\n\n"
    else:
        prompt = f"User: {query}\n\n"

    return prompt, all_urls
