from typing import Dict, List, Optional, Tuple, Union

import torch

from .model_tokenizer import chat_model_tokenizer
from .prompt_generator import prompt_with_rag


def chatbot(
    user_query: str, history: str, use_google: bool, search_time: str
) -> str:
    """
    Function to simulate a chatbot conversation.

    Parameters:
    - user_query (str): The user's input query to the chatbot.
    - history (List[str]): The conversation history.

    Returns:
    - A response to the chat_history+user_query.
    """
    user_query = str(user_query)
    search_time = str(search_time)

    new_user_query, all_urls = prompt_with_rag(user_query, use_google, search_time)

    new_user_query = (
        "You are an expert financial ChatBot, respond to the user message and feel"
        " free to use the extra given online source information during the"
        " conversation, if necessary.\n\n"
        + new_user_query
    )


    model, tokenizer = chat_model_tokenizer()
    with torch.no_grad():
        response, history = model.chat(tokenizer, new_user_query, history=history)
    
    # print("=" * 5)
    # print("new_user_query: ", new_user_query)
    # print("=" * 5)
    # print("response: ", response)
    # print("=" * 5)
    if use_google:
        return "{}\n\nReferences:\n{}".format(response, "\n".join(all_urls))
    else:
        return response


# def chatbot2(model_name: str, user_query: str) -> Tuple[str, List[str]]:
#     """
#     Function to simulate a chatbot conversation.

#     Parameters:
#     - model_name (str): The name or path of the pretrained language model.
#     - user_query (str): The user's input query to the chatbot.

#     Returns:
#     - Tuple[str, List[str]]: A tuple containing the model's response and the updated conversation history.
#     """

#     # Check if CUDA (GPU) is available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load model and tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

#     # Initialize conversation history
#     conversation_history = []

#     # Add user input to conversation history
#     conversation_history.append(">> User: " + user_query + "\n" + ">> Bot: ")

#     # Tokenize and generate model response
#     input_ids = tokenizer.encode(
#         "\n".join(conversation_history), return_tensors="pt"
#     ).to(device)
#     with torch.no_grad():
#         output = model.generate(
#             input_ids,
#             max_length=500,
#             num_beams=5,
#             no_repeat_ngram_size=2,
#             top_k=50,
#             top_p=0.95,
#             temperature=0.7,
#         )
#     model_response = tokenizer.decode(output[0], skip_special_tokens=True)

#     # Add model response to conversation history
#     conversation_history.append(">> Bot: " + model_response)

#     return model_response, conversation_history
