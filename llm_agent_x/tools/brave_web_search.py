from llm_agent_x.constants import redis_port


import redis
import requests
from bs4 import BeautifulSoup
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.utils import get_stop_words


import hashlib
import json
import math
import time
from os import getenv
from typing import Dict, List

from llm_agent_x.constants import LANGUAGE, redis_db, redis_expiry, redis_host


def brave_web_search(query: str, num_results: int) -> List[Dict[str, str]]:
    """
    Perform a web search with the given query and number of results using the Brave Search API, returning JSON-formatted results. 
    Make sure to be very specific in your search phrase. Ask for specific information, not general information.

    :param query: The search query.
    :param num_results: The desired number of results. This will be passed as the 'count' parameter to the Brave API.
    :return: A JSON-formatted list of dictionaries containing the search results (title, url, content),
             or an empty list if the API call fails, returns an error, or no results are found.
    """



    try:
        r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        r.ping()  # Check connection
        print("Redis connection successful.")
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection failed: {e}.  Caching disabled.")
        r = None  # Disable caching if connection fails

    # --- 1. Cache Key Generation ---
    cache_key = hashlib.md5(f"{query}_{num_results}".encode("utf-8")).hexdigest()

    # --- 2. Attempt Cache Retrieval ---
    if r:
        cached_result = r.get(cache_key)
        if cached_result:
            print("Cache hit!")
            try:
                return json.loads(cached_result.decode("utf-8"))
            except json.JSONDecodeError:
                print("Error decoding JSON from cache.  Ignoring cached result.")
                pass  # Fall through to API call
    # Some constants for request handling
    SCRAPE_TIMEOUT_SECONDS = 10
    max_scrape_chars = 5000  # Example max character count for scrape
    REQUEST_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    def get_page_text_content(element):
        """
        Extract and normalize text content from an HTML element.

        :param element: The HTML element to extract content from.
        :return: A string containing the cleaned-up text content of the element.
        """
        return element.get_text(" ", strip=True)

    api_key = getenv("BRAVE_API_KEY")
    if not api_key or api_key == "YOUR_ACTUAL_BRAVE_API_KEY_HERE":
        print("BRAVE_API_KEY environment variable not set or is a placeholder.")
        return []  # Return empty list instead of JSON error

    base_url = "https://api.search.brave.com/res/v1/web/search"
    brave_headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params = {"q": query, "count": num_results}

    response_obj = None

    max_retries = 3  # Maximum number of retries for 429 errors
    retry_wait_time = 10  # Wait time in seconds before retrying on a 429 error
    retry_count = 0

    while retry_count <= max_retries:
        try:
            # --- 1. Brave Search API Call ---
            search_response = requests.get(
                base_url, headers=brave_headers, params=params, timeout=10
            )
            response_obj = search_response

            # Handle 429 Too Many Requests
            if search_response.status_code == 429:
                retry_count += 1
                if retry_count <= max_retries:
                    print(
                        f"Rate limit reached. Retrying in {retry_wait_time} seconds... (Retry {retry_count}/{max_retries})"
                    )
                    time.sleep(retry_wait_time)
                    continue  # Retry the request
                else:
                    print("Rate limit exceeded. Exhausted all retries.")
                    return [] # Return empty list instead of JSON error

            search_response.raise_for_status()
            json_response_data = search_response.json()

            extracted_results_for_llm = []

            if json_response_data.get("web") and json_response_data["web"].get(
                "results"
            ):
                search_results_to_process = json_response_data["web"]["results"]

                for i, result in enumerate(search_results_to_process):
                    title = result.get("title")
                    url = result.get("url")
                    snippet = result.get(
                        "description", ""
                    )  # Default to empty string if no description

                    content_for_llm = snippet  # Default to snippet

                    if title and url:  # We need a URL to attempt scraping
                        print(
                            f"  Processing result {i+1}/{len(search_results_to_process)}: '{title}' ({url})"
                        )
                        try:
                            # --- 2. Scrape Individual Webpage ---
                            print(f"    Attempting to scrape content from {url}...")
                            page_response = requests.get(
                                url,
                                headers=REQUEST_HEADERS,
                                timeout=SCRAPE_TIMEOUT_SECONDS,
                                allow_redirects=True,
                            )
                            page_response.raise_for_status()  # Check for HTTP errors for the page itself

                            # Ensure content type is HTML before parsing
                            content_type = page_response.headers.get(
                                "Content-Type", ""
                            ).lower()
                            if "text/html" in content_type:
                                # Using lxml is generally faster and more robust
                                soup = BeautifulSoup(
                                    page_response.content, "lxml"
                                )  # or 'html.parser'

                                # More targeted text extraction (example, adjust as needed)
                                # Try to find common main content containers
                                main_content_text = ""
                                main_content_elements = soup.find_all(
                                    ["article", "main"]
                                )
                                if main_content_elements:
                                    for el in main_content_elements:
                                        main_content_text += (
                                            get_page_text_content(el) + " "
                                        )
                                    main_content_text = main_content_text.strip()

                                if (
                                    not main_content_text
                                ):  # Fallback to body if no specific main content found
                                    if soup.body:
                                        main_content_text = get_page_text_content(
                                            soup.body
                                        )
                                    else:  # Fallback if no body tag (highly unlikely for valid HTML)
                                        main_content_text = get_page_text_content(soup)

                                if main_content_text:
                                    print(
                                        f"    Scraped {len(main_content_text)} chars. Max allowed: {max_scrape_chars}"
                                    )
                                    if 0 < len(main_content_text) <= max_scrape_chars:
                                        content_for_llm = main_content_text
                                        print(
                                            f"    Using scraped content (length: {len(main_content_text)})."
                                        )
                                    elif len(main_content_text) > max_scrape_chars:
                                        print(
                                            f"    Scraped content too long ({len(main_content_text)} chars), summarizing."
                                        )
                                        # content_for_llm = (
                                        #     snippet
                                        #     + " [Note: Full content exceeded character limit]"
                                        # )
                                        SENTENCES_COUNT = math.floor(len(main_content_text) / 150)

                                        parser = PlaintextParser.from_string(main_content_text, Tokenizer(LANGUAGE))
                                        stemmer = Stemmer(LANGUAGE)

                                        summarizer = Summarizer(stemmer)
                                        summarizer.stop_words = get_stop_words(LANGUAGE)

                                        content_for_llm = " ".join(summarizer(parser.document, SENTENCES_COUNT))

                                    else:  # Scraped text was empty
                                        print(
                                            f"    Scraped content was empty, using snippet."
                                        )
                                else:
                                    print(
                                        f"    Could not extract meaningful text, using snippet."
                                    )
                            else:
                                print(
                                    f"    Skipping scrape: Content-Type is '{content_type}', not HTML. Using snippet."
                                )
                                content_for_llm = (
                                    snippet + " [Note: Content was not HTML]"
                                )

                        except requests.exceptions.Timeout:
                            print(f"    Scraping timed out for {url}. Using snippet.")
                            content_for_llm = (
                                snippet + " [Note: Page timed out during scraping]"
                            )
                        except requests.exceptions.HTTPError as e:
                            print(
                                f"    HTTP error {e.response.status_code} while scraping {url}. Using snippet."
                            )
                            content_for_llm = (
                                snippet
                                + f" [Note: HTTP {e.response.status_code} during scraping]"
                            )
                        except requests.exceptions.RequestException as e:
                            print(f"    Error scraping {url}: {e}. Using snippet.")
                            content_for_llm = snippet + " [Note: Error during scraping]"
                        except (
                            Exception
                        ) as e:  # Catch-all for any other scraping/parsing errors
                            print(
                                f"    Unexpected error scraping/parsing {url}: {e}. Using snippet."
                            )
                            content_for_llm = (
                                snippet + " [Note: Unexpected error during scraping]"
                            )

                        extracted_results_for_llm.append(
                            {
                                "title": title,
                                "url": url,
                                "content": content_for_llm.strip(),  # Ensure no leading/trailing ws
                            }
                        )
                    else:  # If no title or URL from Brave search
                        if title and snippet:  # Still add if we have title and snippet
                            extracted_results_for_llm.append(
                                {
                                    "title": title,
                                    "url": url or "N/A",
                                    "content": snippet.strip(),
                                }
                            )

                # --- 4. Cache the Result (if successful) ---
                if r and extracted_results_for_llm:
                    try:
                        r.setex(
                            cache_key,
                            redis_expiry,
                            json.dumps(extracted_results_for_llm),
                        )
                        print("Result cached in Redis.")
                    except Exception as e:
                        print(f"Error caching result: {e}")

                return extracted_results_for_llm

            if "errors" in json_response_data:
                print(
                    "API returned errors in response",
                )
                return []
            if not extracted_results_for_llm:
                return []  # No web results found or processed

        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            return [] #Return empty list instead of JSON error


    return [] # Return empty list if all retries fail