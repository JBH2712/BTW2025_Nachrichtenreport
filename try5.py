import openai
import os
import time
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import chromedriver_autoinstaller


### Webscraping Functions:

# Part 1: Fetch news articles and store them in a DataFrame
def get_news(api_key, query, language='en', from_date=None, to_date=None, domains=None, page=1, pageSize=100, sort_by='publishedAt'):
    """
    Fetch news articles from the NewsAPI.

    Parameters:
        api_key (str): Your NewsAPI key.
        query (str): The search query for the articles.
        language (str): Language of the articles (default is 'en').
        from_date (str): Start date for the articles in YYYY-MM-DD format.
        to_date (str): End date for the articles in YYYY-MM-DD format.
        domains (str): Comma-separated list of domains to include in the results.
        page (int): Page number for pagination (default is 1).

    Returns:
        pd.DataFrame: DataFrame containing article data.
    """
    url = "https://newsapi.org/v2/everything"
    
    # Define parameters for the API request
    params = {
        'q': query,
        'language': language,
        'from': from_date,
        'to': to_date,
        'sortBy': sort_by,
        'domains': domains,
        'page': page,
        'pageSize': pageSize,
        'apiKey': api_key
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        articles_data = response.json().get("articles", [])
        articles_list = []

        for article in articles_data:
            articles_list.append({
                "Title": article.get('title'),
                "Source": article.get('source', {}).get('name'),
                "Published At": article.get('publishedAt'),
                "URL": article.get('url')
            })

        return pd.DataFrame(articles_list)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()

# Part 2: Scrape article content and add it to the DataFrame
def scrape_with_selenium(url):
    """
    Scrape the content of a news article from the given URL.
    """
    # Automatically install ChromeDriver
    chromedriver_autoinstaller.install()

    # Set Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    # Initialize WebDriver
    service = Service()

    driver = webdriver.Chrome(service=service, options=chrome_options)
    try:
        driver.get(url)
        time.sleep(5)  # Wait for JavaScript to load completely

        # Extract the page source
        page_source = driver.page_source

        # Use BeautifulSoup to parse the HTML content
        soup = BeautifulSoup(page_source, 'html.parser')

        # Extract the main content of the article (modify selectors as needed)
        article_content = soup.find_all('p')  # Example: Extract all paragraphs

        # Combine the extracted text
        full_text = "\n".join([p.get_text() for p in article_content])

        return full_text

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

    finally:
        driver.quit()
