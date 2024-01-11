"""
This module provides functions for loading chat-based language models and text summarization models.
"""

from typing import Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# To prevent loading the same model multiple times, we use global variables.
chat_model = None
chat_tokenizer = None
summarize_model = None
summarize_tokenizer = None


def chat_model_tokenizer(
    model_name: str = "Qwen/Qwen-14B-Chat-Int4",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a chat-based language model and tokenizer.

    Parameters:
        model_name (str): The name of the pre-trained language model.

    Returns:
        The loaded language model and tokenizer for the chatbot.
    """
    global chat_model, chat_tokenizer
    if chat_model is None or chat_tokenizer is None:
        chat_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        chat_model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True
        ).eval()
    return chat_model, chat_tokenizer


def text_summarization(
    model_name: str = "Falconsai/text_summarization",
) -> Tuple[pipeline, None]:
    """
    Load a text summarization model and tokenizer.

    Parameters:
        model_name (str): The name of the pre-trained summarization model.

    Returns:
        The loaded summarization model and tokenizer.
    """
    global summarize_model, summarize_tokenizer

    if summarize_model is None:
        summarize_model = pipeline("summarization", model=model_name, device_map="auto")
        summarize_tokenizer = None  # No need

    return summarize_model, summarize_tokenizer
