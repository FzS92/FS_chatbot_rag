"""
In this file, we will store all the deprecated models that we have used in the past.
All the models that are currently used are in other src files.
"""


## summarization model

# We can also use the following code to load a financial summarization model but the current used model is better!
# from transformers import PegasusForConditionalGeneration,; PegasusTokenizer,; TFPegasusForConditionalGeneration
# def finance_summarize_model_tokenizer():
#     # Load the model and the tokenizer
#     global summarize_model, summarize_tokenizer

#     model_name = "human-centered-summarization/financial-summarization-pegasus"

#     if summarize_tokenizer is None or summarize_model is None:
#         summarize_tokenizer = PegasusTokenizer.from_pretrained(
#             model_name
#         )  # padding='max_length', truncation=True, max_length=512)
#         summarize_model = PegasusForConditionalGeneration.from_pretrained(
#             model_name
#         ).eval()

#     return summarize_model, summarize_tokenizer


# from .model_tokenizer import finance_summarize_model_tokenizer
# def generate_summary(
#     text: str, max_length: int = 20, num_beams: int = 5, early_stopping: bool = True
# ) -> str:
#     # Load the model and the tokenizer
#     model, tokenizer = finance_summarize_model_tokenizer()

#     # Tokenize the input text
#     input_ids = tokenizer(text, return_tensors="pt").input_ids
#     if input_ids.shape[1] > 512:
#         # input_ids = torch.cat((input_ids[:, :511], input_ids[:, -1:]), dim=1)
#         input_ids = input_ids[:, :512]

#     # Generate the output using the model
#     output = model.generate(
#         input_ids,
#         max_length=max_length,
#         num_beams=num_beams,
#         early_stopping=early_stopping,
#     )

#     # Decode and return the generated summary
#     return tokenizer.decode(output[0], skip_special_tokens=True)


## A model for chat

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


## Web text extraction
# import requests
# from bs4 import BeautifulSoup

# def get_website_text(url):
#     try:
#         response = requests.get(url)
#         soup = BeautifulSoup(response.text, "html.parser")
#         text = soup.get_text()
#         return text
#     except Exception as e:
#         print(f"Error fetching text from {url}: {e}")
#         return None


## Summary using different strategies

# def generate_summary(
#     text: str, max_length: int = 20, num_beams: int = 5, early_stopping: bool = True
# ) -> str:
#     """
#     Generate a summary using the Pegasus model.

#     Parameters:
#         text (str): The input text to be summarized.
#         max_length (int, optional): The maximum length of the generated summary. Defaults to 32.
#         num_beams (int, optional): The number of beams for beam search. Defaults to 5.
#         early_stopping (bool, optional): Whether to stop generation when at least one beam has finished. Defaults to True.

#     Returns:
#         str: The generated summary.
#     """
#     # Load the model and the tokenizer
#     model, tokenizer = finance_summarize_model_tokenizer()

#     # Tokenize the input text
#     input_ids = tokenizer(text, return_tensors="pt").input_ids
#     if input_ids.shape[1] > 512:
#         # input_ids = torch.cat((input_ids[:, :511], input_ids[:, -1:]), dim=1)
#         input_ids = input_ids[:, :512]

#     # Generate the output using the model
#     output = model.generate(
#         input_ids,
#         max_length=max_length,
#         num_beams=num_beams,
#         early_stopping=early_stopping,
#     )

#     # Decode and return the generated summary
#     return tokenizer.decode(output[0], skip_special_tokens=True)


# def generate_summary(text: str) -> str:
#     """
#     Function to summarize.

#     Parameters:
#     - model_name (str): The name or path of the pretrained language model.
#     - user_query (str): The user's input query to the chatbot.

#     Returns:
#     - Tuple[str, List[str]]: A tuple containing the model's response and the updated conversation history.
#     """

#     # Add "summarize: " prefix to the user query
#     text = f"Summarize: {text}"

#     model, tokenizer = chat_model_tokenizer()
#     response, _ = model.chat(tokenizer, text, history=[])

#     return response
