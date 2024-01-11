from typing import Dict, List, Optional, Tuple, Union

import torch

from .model_tokenizer import chat_model_tokenizer
from .prompt_generator import prompt_with_rag


def chatbot(user_query: str, history: List, use_google: bool, search_time: str) -> str:
    """
    Function to simulate a chatbot conversation.

    Parameters:
    - user_query (str): The user's input query to the chatbot.
    - history (List[str]): The conversation history.
    - use_google (bool): Whether to use Google search for the next message.
    - search_time (str): The time frame for Google search.

    Returns:
    - A response to the chat_history+user_query.
    """

    # Bytestring to strings conversion
    user_query = str(user_query)
    search_time = str(search_time)

    try:
        new_user_query, all_urls = prompt_with_rag(user_query, use_google, search_time)
    except Exception as e:
        return f"Error generating prompt: {str(e)}"

    # Prompt engineering for the financial chatbot
    new_user_query = (
        "You are an expert financial ChatBot, respond to the user message and feel"
        " free to use the extra given online source information during the"
        " conversation, if necessary.\n\n"
        + new_user_query
    )

    model, tokenizer = chat_model_tokenizer()
    with torch.no_grad():
        response, history = model.chat(tokenizer, new_user_query, history=history)

    if use_google:
        return "{}\n\nReferences:\n{}".format(response, "\n".join(all_urls))
    else:
        return response
