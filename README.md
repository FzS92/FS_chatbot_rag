# Financial Chatbot with Retrieval-Augmented Generation (RAG)

This is a Python-based financial chatbot equipped with **Retrieval-Augmented Generation (RAG)**. The chatbot seamlessly interacts with users, addressing financial queries by combining the top insights from online sources with the user's input.

## Overview

![ScreenShot](./photos/app.png)


The financial chatbot follows the following workflow to ensure efficiency, accuracy and currency of information:

1. **User Query**: Receive and process the user's financial question.
2. **Summarization**: Summarize lengthy user prompts for an enhanced online search experience.
3. **Google Search**: Perform a Google search to extract the top 3 results within a specific time frame.
4. **Web Scraping**: Utilize Selenium to extract text content from the searched websites (Chrome installation required).
5. **Text Refinement**: Eliminate short paragraphs, duplicates, and unnecessary new lines to ensure superior information quality from web scraping.
6. **Semantic Search**: Employ the robust 'all-mpnet-base-v2' model to encode the user query and online sources, thereby enhancing semantic search capabilities. Utilize the most relevant parts of the new sources.
7. **Summarization (Again)**: Independently summarize top-related paragraphs to prevent information loss and optimize the responsiveness of the chatbot.
8. **Prompt Engineering**: Combine the user query and summarized paragraphs using prompt engineering.
9. **Qwen Model**: Leverage the high-quality *"Qwen/Qwen-14B-Chat-Int4"* model for response generation.
10. **Response and References**: Provide the user with the generated response and references.

## Usage

To employ the financial chatbot, follow these simple steps:


1. Clone the repository and go to the downloaded file.
```python
git clone https://github.com/FzS92/FS_chatbot_rag.git
cd FS_chatbot_rag
```


2. Create a new enviornment and install the required dependencies:
```python
conda create -n fschat python=3.8
conda activate fschat
pip install -r requirements.txt
```

Optional: To enhance efficiency and reduce memory usage, you can install the following dependencies:

```python
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
pip install csrc/layer_norm
pip install csrc/rotary
```


3. Run the `main.py` script.

```python
python main.py
```

You will see some links in the terminal. Click on the first link to launch the Gradio ChatInterface. Pose financial questions and witness the chatbot responding with the top online information!