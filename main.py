"""
This script launches a Gradio ChatInterface for a financial chatbot. The chatbot
utilizes various methods for chatbot simulation, text summarization, web scraping, 
    and semantic search.

Usage:
- Run this script to launch the Gradio interface for the financial chatbot.
- Users can ask financial questions, and the chatbot responds based on predefined models, 
    Google search, semantic search, and summarization.

Methods:
- `main`: Initializes and launches the Gradio ChatInterface.
- `chatbot`: Responds ta the user's message considering chat history and online sources.
- `chat_model_tokenizer`: Load a chat-based language model and tokenizer (Qwen/Qwen-14B-Chat-Int4).
- `text_summarization`: Load a text summarization model and tokenizer (Falconsai/text_summarization).
- `get_website_text`: Extracts text content from a given website using Selenium.
- `google_search`: Perform a Google search for the user's query or its summarization (if it was long)
        and save the URLs for specified number of results.
- `summarize_text`: Summarizes the top related information and use them in the prompt as retrival information.
- `prompt_with_rag`: Get information from online sources related to the given query and combine them.
- `filter_strings_by_word_count`: Removes paragraphs with 22 words or less.
- `separate_paragraphs`: Separates a text into unique paragraphs.
- `remove_extra_newlines`: Removes extra newlines from a string.
- `encode_text`: Encode the given list of texts using the specified model.
- `get_top_n_cosine_similarity_rows`: Get the indices of the top N rows with highest 
        cosine similarity scores.
- `semantic_search`: Perform semantic search by measuring the similarity between encoded of 
        paragraphs or sentences in a corpus and the query.

Author:
- Farzad Salajegheh, 2024
"""

import gradio as gr

from src import chatbot


class FinancialChatbotApp:
    def __init__(self):
        self.chatbot = chatbot

    def main(self):
        iface = gr.ChatInterface(
            title="Financial Chatbot with Retrieval-Augmented Generation (RAG)",
            fn=self.chatbot,
            chatbot=gr.Chatbot(height=750),
            textbox=gr.Textbox(
                placeholder="Ask me a question", container=False, scale=7
            ),
            description="Ask financial questions!",
            theme="soft",
            # examples=["What is the price of S&P", "Am I cool?", "Are tomatoes vegetables?"],
            # cache_examples=False,
            retry_btn=None,
            undo_btn="Delete Previous",
            clear_btn="Clear",
            additional_inputs=[
                gr.Checkbox(label="Google search for the next message?", value=True),
                gr.Dropdown(
                    choices=["All", "Year", "Month", "Week", "Day", "Hour"],
                    label="Search Time Frame",
                    value="All",
                ),
            ],
            additional_inputs_accordion=gr.Accordion(
                open=True, label="Do you want to use online sources?"
            ),
        ).launch(share=False)


if __name__ == "__main__":
    app = FinancialChatbotApp()
    app.main()
