"""
This script provides a function for obtaining information from online sources related
to a given query.
The `prompt_with_rag` function utilizes online search and semantic search to retrieve
relevant information.
"""

from typing import List, Tuple

from .online_search import get_website_text, google_search
from .semantic_search import semantic_search


def prompt_with_rag(
    query: str, use_google: bool, search_time: str, num_results: int = 3
) -> Tuple[str, List]:
    """
    Get information from online sources related to the given query.

    Parameters:
    - query (str): The user's query.
    - use_google (bool): Whether to use Google search.
    - search_time (str): The time range for the search.
    - num_results (int): The maximum number of online search results to consider.

    Returns:
    - Tuple[str, List[str]]: A tuple containing prompt and a list of URLs.
    """
    max_prompt_length = 500  # To avoid very long prompts

    all_urls = ""
    answer = ""
    loop_count = 0
    if use_google:
        all_urls = google_search(query, search_time)
        num_results = min(
            num_results, len(all_urls)
        )  # Making sure that we got some results

        while len(answer.split()) < max_prompt_length and loop_count < num_results:
            url_in_use = all_urls[loop_count]
            top_text = get_website_text(url_in_use)

            # Get the top answers ans also summarize them
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
