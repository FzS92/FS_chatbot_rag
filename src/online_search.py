"""
This script provides functions for extracting text content from websites using Selenium and
performing Google searches to retrieve URLs based on a specified query.

Functions:
- get_website_text: Extracts text content from a given website using Selenium.
- google_search(query, num_results=3): Performs a Google search and retrieves URLs for the
  specified number of results. Considering the time frame.
"""

import logging
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

# logging configuration
logging.basicConfig(level=logging.INFO)


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
    # No UI, to run on server.
    # (A very small number of websites may block headless browsers)
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")  # No UI, to run on server
    chrome_options.add_argument(
        "--blink-settings=imagesEnabled=false"
    )  # Not images, only text & faster
    chrome_options.add_argument("--disable-javascript")  # No JS (faster)
    chrome_options.add_argument("--disable-plugins")  # No plugins (faster)
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
        logging.info("URL is: %s", url)
        driver.get(url)

        try:
            # Wait for the page to load
            WebDriverWait(driver, max_wait_time).until(
                EC.presence_of_element_located((By.XPATH, "/html/body"))
            )
        except TimeoutException:
            logging.warning("Page load timed out for URL: %s", url)

        # Extract the text from the entire page
        page_text = driver.find_element(By.XPATH, "/html/body").text  # type: WebElement

        return page_text


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
    max_user_query_len = 10
    # Check if the number of words in user's quey is more than max_user_query_len
    if len(query.split()) > max_user_query_len:
        # If more than max_user_query_len, summarize it.
        logging.info("Query to send before sending to Google:\n%s", query)
        query = summarize_text(query)
        logging.info("Query to Google after summarization: %s", query)

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
    For better Google search, min length can be in range of 1 to 3.
    But should not be very long. (chosen 25 by experiments)

    Args:
        text (str): The input text to be summarized.
        max_length (int): The maximum length of the summary. Defaults to 25.
        min_length (int): The minimum length of the summary. Defaults to 3.
        do_sample (bool): If True, uses sampling to generate the summary.

    Returns:
        str: The summarized text.
    """
    model, _ = text_summarization()
    summary = model(
        text, max_length=max_length, min_length=min_length, do_sample=do_sample
    )
    return summary[0]["summary_text"]
