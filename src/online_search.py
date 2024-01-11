"""
This script provides functions for extracting text content from websites using Selenium and
performing Google searches to retrieve URLs based on a specified query.

Functions:
- get_website_text(url, chrome_driver_path=None, max_wait_time=10): Extracts text content
  from a given website using Selenium.
- google_search(query, num_results=3): Performs a Google search and retrieves URLs for the
  specified number of results.
"""

from typing import List

from googlesearch import search
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from .model_tokenizer import text_summarization

# from .model_tokenizer import finance_summarize_model_tokenizer

# import requests
# from bs4 import BeautifulSoup


def get_website_text(
    url: str, chrome_driver_path: str = None, max_wait_time: int = 10
) -> str:
    """
    Extracts text content from a given website using Selenium.

    Parameters:
    - url (str): The URL of the website to extract text from.
    - chrome_driver_path (str, optional): Path to the Chrome driver executable. If not provided,
      Selenium will attempt to find the driver in the system's PATH.
    - max_wait_time (int, optional): Maximum time to wait for the page to load, in seconds.

    Returns:
    - str: The extracted text content from the entire page.
    """
    # Set up Chrome options and service to extract text only
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")  # No UI
    chrome_options.add_argument("--blink-settings=imagesEnabled=false")  # Not images
    chrome_options.add_argument("--disable-javascript")  # No JS
    chrome_options.add_argument("--disable-plugins")  # No plugins
    chrome_service = (
        ChromeService(executable_path=chrome_driver_path)
        if chrome_driver_path
        else None
    )

    # Initialize Chrome WebDriver
    with webdriver.Chrome(
        service=chrome_service, options=chrome_options
    ) as driver:  # type: WebDriver
        # Set the maximum wait time
        driver.implicitly_wait(max_wait_time)

        # Open the website
        print("URL is: " + url)
        driver.get(url)

        try:
            # Wait for the page to load
            WebDriverWait(driver, max_wait_time).until(
                EC.presence_of_element_located((By.XPATH, "/html/body"))
            )
        except TimeoutException:
            pass

        # Extract the text from the entire page
        page_text = driver.find_element(By.XPATH, "/html/body").text  # type: WebElement

        return page_text


# def get_website_text(url):
#     try:
#         response = requests.get(url)
#         soup = BeautifulSoup(response.text, "html.parser")
#         text = soup.get_text()
#         return text
#     except Exception as e:
#         print(f"Error fetching text from {url}: {e}")
#         return None


def google_search(
    query: str, search_time: str, num_results: int = 3, lang: str = "en"
) -> List[str]:
    """
    Perform a Google search and save the URLs for specified number of results.

    Parameters:
    - query (str): The search query to be used.
    - num_results (int, optional): The number of search results to retrieve (default is 3).
    - lang (str, optional): The language of the search results (default is 'en' for English).

    Returns:
    List[str]: A list containing the search results.
    """

    # Check if the number of words is more than 10
    if len(query.split()) > 10:
        # If more than 10 words, feed it to the generate_summary method
        print("Qury to send before sending to Google:\n" + query)
        query = summarize_text(query)
        print("\nQuery to Google after summarization: " + query)

    dic_time = {
        "All": "a",
        "Year": "y",
        "Month": "m",
        "Week": "w",
        "Day": "d",
        "Hour": "h",
    }

    search_results = []
    for result in search(
        query,
        num=num_results,
        stop=num_results,
        pause=3,
        lang=lang,
        tld="com",
        tbs=f"qdr:{dic_time[search_time]}",
    ):
        search_results.append(result)
    return search_results


def summarize_text(
    text: str, max_length: int = 25, min_length: int = 3, do_sample: bool = False
) -> str:
    """
    Summarizes the input text using the Hugging Face summarization pipeline.

    Args:
        text (str): The input text to be summarized.
        max_length (int, optional): The maximum length of the summary. Defaults to 25.
        min_length (int, optional): The minimum length of the summary. Defaults to 3.
        do_sample (bool, optional): If True, uses sampling to generate the summary. Defaults to False.

    Returns:
        str: The summarized text.
    """
    model, _ = text_summarization()
    summary = model(
        text, max_length=max_length, min_length=min_length, do_sample=do_sample
    )
    return summary[0]["summary_text"]


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
