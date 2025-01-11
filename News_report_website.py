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
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    service = Service("chromedriver.exe")

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




### Agent Calling Functions

import openai
import time

# Function to get assistant response
def get_assistant_response(assistant_id, message_content, api_key):
    # Set OpenAI API key
    openai.api_key = api_key

    # Create a new thread
    thread = openai.beta.threads.create()

    # Add a user message to the thread
    openai.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message_content
    )

    # Create a run for the assistant
    run = openai.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id
    )

    # Wait for the run to complete
    while run.status in ["queued", "in_progress"]:
        time.sleep(0.5)
        run = openai.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

    # Retrieve messages from the thread
    messages = openai.beta.threads.messages.list(thread_id=thread.id)
    assistant_response = messages.data[0].content[0].text.value

    return assistant_response









st.title("Bundestagswahl 2025 Analysis")

# Sidebar inputs for API keys
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""
if "newsapi_key" not in st.session_state:
    st.session_state["newsapi_key"] = ""

# Input fields for API keys
with st.sidebar:
    st.session_state["openai_api_key"] = st.text_input(
        "OpenAI API-Schlüssel eingeben:",
        type="password",
        placeholder="OpenAI API-Schlüssel hier eingeben"
    )
    st.session_state["newsapi_key"] = st.text_input(
        "NewsAPI API-Schlüssel eingeben:",
        type="password",
        placeholder="NewsAPI API-Schlüssel hier eingeben"
    )
    if st.session_state["openai_api_key"] and st.session_state["newsapi_key"]:
        st.success("Beide API-Schlüssel wurden gespeichert!")

# Initialize session state to store outputs
if "background" not in st.session_state:
    st.session_state.background = None
if "title_summary" not in st.session_state:
    st.session_state.title_summary = None
if "articles_summary" not in st.session_state:
    st.session_state.articles_summary = None
# Initialize session state to store the article
if "Article" not in st.session_state:
    st.session_state.Article = None


if not st.session_state.Article:
    st.write("""Bleiben Sie informiert: Ihr persönlicher Nachrichtenreport zur Bundestagswahl 2025

Erhalten Sie einen maßgeschneiderten Überblick über die wichtigsten Entwicklungen und Nachrichten zur Bundestagswahl 2025 – auf Knopfdruck! Unsere Website analysiert und sammelt innerhalb von Minuten die relevantesten Nachrichten der letzten 24 Stunden.

Egal, ob Sie Wähler, Journalist oder politisch interessiert sind: Mit unserem automatisierten Report verpassen Sie keine Schlagzeilen und sind bestens informiert. Effizient, präzise und immer auf dem neuesten Stand – probieren Sie es jetzt aus!
""")

# Button to generate the article
if st.button("Persönlichen Nachrichtenreport erstellen"):

    # Step 1: Fetch the webpage
    url = "https://www.wahlrecht.de/umfragen/forsa.htm"
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Step 2: Parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")

    # Step 3: Locate the table (using its class "wilko")
    table = soup.find("table", class_="wilko")

    # Step 4: Extract table headers
    headers = []
    for th in table.find("thead").find_all("th"):
        headers.append(th.get_text(strip=True))

    # Step 5: Extract table rows
    data = []
    rows = table.find("tbody").find_all("tr")
    for row in rows:
        cells = row.find_all(["th", "td"])
        row_data = [cell.get_text(strip=True) for cell in cells]
        data.append(row_data)

    # Step 6: Create a DataFrame
    dataframe = pd.DataFrame(data, columns=headers)

    # Step 7: Save the DataFrame (optional)
    dataframe.to_csv("poll_data.csv", index=False, sep=';')

    # Step 8: Clean the Data
    # Load the dataset with a semicolon delimiter
    data = pd.read_csv("poll_data.csv", delimiter=';')

    # 1. Rename the first column to "Datum"
    data.rename(columns={data.columns[0]: "Datum"}, inplace=True)

    # 2. Drop unwanted columns
    data.drop(data.columns[[1, 11]], axis=1, inplace=True)  # Drop second and 12th columns by position
    data.drop(columns=["Nichtwähler/Unentschl.", "Befragte", "Zeitraum"], inplace=True)

    # 3. Convert the "Datum" column to datetime format
    data["Datum"] = pd.to_datetime(data["Datum"], format='%d.%m.%Y', errors='coerce')

    # Remove rows where the date in the "Datum" column is older than "2024-11-06"
    data = data[data["Datum"] >= pd.Timestamp("2024-11-06")]

    # 4. Clean remaining columns
    for column in data.columns[1:]:  # Skip the "Datum" column
        data[column] = (
            data[column]
            .str.replace('%', '', regex=False)  # Remove the % symbol
            .str.replace(' ', '', regex=False)  # Remove spaces
            .str.replace(',', '.', regex=False)  # Replace comma with decimal point for float conversion
            .replace('–', '0')  # Replace "-" with 0
            .astype(float)  # Convert to float for numerical operations
        )

    # Save the cleaned dataset
    data.to_csv('cleaned_poll_data.csv', index=False)

    # Step 9: Visualize the Data
    # Convert the 'Datum' column to datetime format
    data['Datum'] = pd.to_datetime(data['Datum'])

    # Set the 'Datum' column as the index
    data.set_index('Datum', inplace=True)

    # Define colors for each party
    party_colors = {
        'CDU/CSU': 'black',
        'SPD': 'red',
        'GRÜNE': 'green',
        'FDP': 'yellow',
        'LINKE': 'magenta',
        'AfD': 'blue',
        'FW': 'orange',
        'BSW': 'purple',
        'Sonstige': 'grey'
    }

    # Plot the development of the parties over time
    plt.figure(figsize=(14, 8))

    # Use the header to identify party columns (all columns except the first one)
    party_columns = data.columns[0:]
    for party in party_columns:
        color = party_colors.get(party, 'black')  # Default to black if the party is not in the dictionary
        plt.plot(data.index, data[party], label=party, color=color)

    plt.title('Development of Parties Over Time')
    plt.xlabel('Date')
    plt.ylabel('Support (%)')
    plt.legend()
    plt.grid(True)

    # Save the graph as a file
    plt.savefig('party_change.png')

    
    # Step 1: Fetch the webpage
    url = "https://www.wahlrecht.de/umfragen/"
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Step 2: Parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")

    # Step 3: Locate the table (using its class "wilko")
    table = soup.find("table", class_="wilko")

    # Step 4: Extract table headers
    headers = []
    for th in table.find("thead").find_all("th"):
        headers.append(th.get_text(strip=True))

    # Step 5: Extract table rows
    data = []
    rows = table.find("tbody").find_all("tr")
    for row in rows:
        cells = row.find_all(["th", "td"])
        data.append([cell.get_text(strip=True) for cell in cells])

    # Step 6: Create a DataFrame
    dataframe = pd.DataFrame(data, columns=headers)

    # Step 7: Clean the data
    # Remove the second and second last columns (assuming they are empty)
    dataframe = dataframe.drop(columns=[dataframe.columns[1], dataframe.columns[-2]])

    # Remove the % sign and replace '-' with 0
    dataframe = dataframe.replace({'%': '', '-': '0'}, regex=True)

    # Remove the last row of the dataset
    dataframe = dataframe.iloc[:-1]

    # Split the dataset into two new datasets
    data_without_last_column = dataframe.iloc[:, :-1]
    last_column_data = dataframe.iloc[:, -1]

    # Calculate the average for each party
    # Extract party names
    party_names = data_without_last_column.iloc[1:, 0]

    # Remove the first row (dates) only from the numerical data
    numerical_data = data_without_last_column.iloc[1:, 1:]

    # Convert data to numeric, replacing commas with dots and handling errors
    numerical_data = numerical_data.applymap(lambda x: str(x).replace(',', '.'))
    numerical_data = numerical_data.apply(pd.to_numeric, errors='coerce')

    # Calculate the average for each party (row)
    row_averages = numerical_data.mean(axis=1)

    # Combine the party names with their averages
    result = pd.DataFrame({"Party": party_names, "Average": row_averages})

    # Visualization
    party_colors = {
        'CDU/CSU': 'black',
        'SPD': 'red',
        'GRÜNE': 'green',
        'FDP': 'yellow',
        'DIE LINKE': 'magenta',
        'AfD': 'blue',
        'FW': 'orange',
        'BSW': 'purple',
        'Sonstige': 'grey'
    }

    colors = [party_colors.get(party, 'grey') for party in result['Party']]

    plt.figure(figsize=(10, 6))
    plt.bar(result['Party'], result['Average'], color=colors, alpha=0.8)
    plt.title('Umfragewerte pro Partei', fontsize=16)
    plt.xlabel('Parteien', fontsize=12)
    plt.ylabel('Prozent', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()

    # Save the plot as an image
    plt.savefig('Viz_Party_Votes.png')
    print("Data processing and visualization completed. Saved as 'Viz_Party_Votes.png'.")


    ### Start the Websraping:

    if __name__ == "__main__":
        # Replace with your NewsAPI key
        API_KEY = st.session_state["newsapi_key"]

        # Define the search query and other parameters
        QUERY = '"Bundestagswahl" AND (Interview OR fordert OR kritisiert OR erklärt OR Aussage OR Position OR Umfrage OR Trend OR Prognose OR Entwicklung OR Einfluss OR Wahlen OR SPD OR CDU OR Grüne OR AfD OR FDP OR Linke) NOT (Sport OR Unterhaltung OR "Ukraine-Krieg im Liveticker:")'
        LANGUAGE = "de"
        PAGE = 1
        SORT_BY = "relevancy"

        # Get current date and adjust the TO_DATE dynamically
        current_date = datetime.now()
        TO_DATE = current_date.strftime("%Y-%m-%d")

        # Optionally calculate a dynamic FROM_DATE (e.g., 7 days before the current date)
        FROM_DATE = (current_date - timedelta(days=2)).strftime("%Y-%m-%d")

        # Fetch news articles into a DataFrame
        titles_df = get_news(api_key=API_KEY, query=QUERY, language=LANGUAGE, from_date=FROM_DATE, to_date=TO_DATE, page=PAGE, sort_by=SORT_BY)

    # print(titles_df)





    if __name__ == "__main__":
        # Replace with your NewsAPI key
        API_KEY = st.session_state["newsapi_key"]

        # Define the search query and other parameters
        QUERY = '"Bundestagswahl" AND (Interview OR fordert OR kritisiert OR erklärt OR Aussage OR Position OR Umfrage OR Trend OR Prognose OR Entwicklung OR Einfluss OR Wahlen OR SPD OR CDU OR Grüne OR AfD OR FDP OR Linke) NOT (Sport OR Unterhaltung OR "Ukraine-Krieg im Liveticker:")'
        LANGUAGE = "de"
        DOMAINS = "welt.de,tagesschau.de,focus.de,spiegel.de,faz.net,sueddeutsche.de,zeit.de" 
        PAGE = 1
        PAGE_SIZE = 25
        SORT_BY = "relevancy"

        # Get current date and adjust the TO_DATE dynamically
        current_date = datetime.now()
        TO_DATE = current_date.strftime("%Y-%m-%d")

        # Optionally calculate a dynamic FROM_DATE (e.g., 7 days before the current date)
        FROM_DATE = (current_date - timedelta(days=2)).strftime("%Y-%m-%d")

        # Fetch news articles into a DataFrame
        articles_df = get_news(API_KEY, QUERY, LANGUAGE, FROM_DATE, TO_DATE, DOMAINS, PAGE, PAGE_SIZE, SORT_BY)

        if not articles_df.empty:
            # Create an empty list to store the results
            scraped_articles = []

            # Loop through each row in the DataFrame
            for index, row in articles_df.iterrows():
                title = row['Title']
                url = row['URL']
                source = row['Source']
                published_at = row['Published At']

                print(f"Scraping URL: {url}")
                article_text = scrape_with_selenium(url)

                scraped_articles.append({
                    "Title": title,
                    "Article": article_text,
                    "Source": source,
                    "Published At": published_at
                })

            # Add the scraped content to the DataFrame
            scraped_df = pd.DataFrame(scraped_articles)

            scraped_df.to_csv('articles.csv', index=False)

            # Display the final DataFrame
            print(scraped_df.head())
        else:
            print("No articles found or an error occurred.")

    # print(scraped_df)
    # print("GAP")
    # print(articles_df)



    # Step 1: Fetch the webpage
    url = "https://www.wahlrecht.de/umfragen/"
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Step 2: Parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")

    # Step 3: Locate the table (using its class "wilko")
    table = soup.find("table", class_="wilko")

    # Step 4: Extract table headers
    headers = []
    for th in table.find("thead").find_all("th"):
        headers.append(th.get_text(strip=True))

    # Step 5: Extract table rows
    data = []
    rows = table.find("tbody").find_all("tr")
    for row in rows:
        cells = row.find_all(["th", "td"])
        data.append([cell.get_text(strip=True) for cell in cells])

    # Step 6: Create a DataFrame
    dataframe = pd.DataFrame(data, columns=headers)

    # print(dataframe)




    poll_data = dataframe

    # Remove the second and second last columns (assuming they are empty)
    poll_data = poll_data.drop(columns=[poll_data.columns[1], poll_data.columns[-2]])

    # Remove the % sign and replace '-' with 0
    poll_data = poll_data.replace({'%': '', '-': '0'}, regex=True)

    # Remove the last row of the dataset
    poll_data_trimmed = poll_data.iloc[:-1]

    # Split the dataset into two new datasets
    # First dataset with all columns except the last one
    data_without_last_column = poll_data_trimmed.iloc[:, :-1]

    # Second dataset with only the last column
    last_column_data = poll_data_trimmed.iloc[:, -1]

    # print(data_without_last_column)


    data = data_without_last_column

    # Extract party names separately to ensure they are preserved
    party_names = data.iloc[1:, 0]

    # Remove the first row (dates) only from the numerical data
    numerical_data = data.iloc[1:, 1:]

    # Convert data to numeric, replacing commas with dots and handling errors
    numerical_data = numerical_data.applymap(lambda x: str(x).replace(',', '.'))
    numerical_data = numerical_data.apply(pd.to_numeric, errors='coerce')

    # Calculate the average for each party (row)
    row_averages = numerical_data.mean(axis=1)

    # Combine the party names with their averages
    current_votes_df = pd.DataFrame({"Party": party_names, "Average": row_averages})

    # print(current_votes)








    ### Call the agents

    # 1. Call the Titel Agent

    # Example usage
    assistant_title_summary = "asst_MR8uZs75kQ6BXjEzCycByeAC"


    if not titles_df.empty:
        titles_df_string = titles_df.to_string(index=False)  # Convert DataFrame to a string without row indices
    else:
        print("No articles were found for the given query.")

    # print(titles_df_string)

    message_title =  f"Das sind die Titel:\n{titles_df_string}"

    title_summary = get_assistant_response(assistant_title_summary, message_title, st.session_state["openai_api_key"])

    # print(title_summary)

    # Generate the article
    st.session_state.title_summary = get_assistant_response(assistant_title_summary, message_title, st.session_state["openai_api_key"])


    # 2. Call the Article Agent

    # Assistant ID for article summary
    assistant_article_summary = "asst_58wjKuiWoq1Yyz1GBpULkdyf"

    # Read the CSV file containing articles
    scraped_articles_df = pd.read_csv("articles.csv", encoding='utf-8')

    # Initialize a dictionary to store the articles
    articles_dict = {}

    # Loop through each row in the dataframe to format articles
    for index, row in scraped_articles_df.iterrows():
        formatted_text = f"Title: {row['Title']}\nSource: {row['Source']}\nArticle: {row['Article']}"
        articles_dict[f"Article_{index + 1}"] = formatted_text

    # Initialize a variable to store combined summaries
    articles_summary = ""

    # Loop through each article in the dictionary
    for article_key, article_content in articles_dict.items():
        # Prepare the message for the assistant
        message_article = f"Der Artikel mit Titel und Quelle: {article_content}"

        # Get the assistant's response
        summary = get_assistant_response(assistant_article_summary, message_article, st.session_state["openai_api_key"])

        # Append the summary to the combined summaries variable
        articles_summary += f"\n{article_key}: {summary}\n"

    # Save the combined summaries to a text file
    with open("articles_summary.txt", "w", encoding="utf-8") as summary_file:
        summary_file.write(articles_summary)

    # print("Summaries saved to articles_summary.txt")

    st.session_state.articles_summary = articles_summary



    # 3. Call the Background Survey Agent

    # Example usage
    assistant_background_summary = "asst_fyhkB5IFkY9gugDc0IJhXnzf"

    if not current_votes_df.empty:
        current_votes_string = current_votes_df.to_string(index=False)  # Convert DataFrame to a string without row indices
    else:
        print("No votes were found for the given query.")

    print(current_votes_string)

    message_background =  f"Das sind die aktuellen Umfragewerte:\n{current_votes_string}"

    background = get_assistant_response(assistant_background_summary, message_background, st.session_state["openai_api_key"])

    # print(background)
    st.session_state.background = get_assistant_response(assistant_background_summary, message_background, st.session_state["openai_api_key"])

    # print(background)
    # print(title_summary)
    # print(articles_summary)


    # 4. Call the Report Writer

    assistant_report_writer = "asst_3EJMFao22u4L5EhVcn5Vv3by"

    message_all_inputs = f"""Das sind die Informationen über die aktuellen Umfragen: {background}
                            Das sind Titel von den Nachrichten des Tages: {title_summary}
                            Das sind Zusammenfassungen zu den Top 25 Artikeln des Tages: {articles_summary}
    """

    Article = get_assistant_response(assistant_report_writer, message_all_inputs, st.session_state["openai_api_key"])








    # Generate the article
    st.session_state.Article = get_assistant_response(assistant_report_writer, message_all_inputs, st.session_state["openai_api_key"])


# Display the article if it has been generated
if st.session_state.Article:
    # Display visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.image("Viz_Party_Votes.png", caption="Umfragewerte pro Partei:", use_column_width=True)
    with col2:
        st.image("party_change.png", caption="Veränderungen in den Umfragen seit dem Bruch der Koalition:", use_column_width=True)

    st.write(st.session_state.Article)
