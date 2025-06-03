import os
import json
import time
import asyncio
import hashlib
from typing import List, Dict

import redis
import httpx
from bs4 import BeautifulSoup

from llm_agent_x.constants import redis_port, redis_db, redis_expiry, redis_host
from llm_agent_x.tools.summarize import summarize


REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def get_page_text_content(element):
    return element.get_text(" ", strip=True)


async def _brave_web_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    # Setup Redis
    try:
        r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        r.ping()
        print("Redis connection successful.")
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection failed: {e}.  Caching disabled.")
        r = None

    cache_key = hashlib.md5(f"{query}_{num_results}".encode("utf-8")).hexdigest()

    if r:
        cached_result = r.get(cache_key)
        if cached_result:
            try:
                print("Cache hit!")
                return json.loads(cached_result.decode("utf-8"))
            except json.JSONDecodeError:
                print("Invalid cache JSON. Ignoring cache.")

    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key or api_key == "YOUR_ACTUAL_BRAVE_API_KEY_HERE":
        print("BRAVE_API_KEY not set correctly.")
        return []

    base_url = "https://api.search.brave.com/res/v1/web/search"
    brave_headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                base_url,
                headers=brave_headers,
                params={"q": query, "count": num_results},
            )
            if response.status_code == 429:
                print("Rate limit exceeded.")
                return []
            response.raise_for_status()
            data = response.json()

            results = data.get("web", {}).get("results", [])
            extracted_results = []

            async def fetch_and_process(result):
                url = result.get("url")
                title = result.get("title")
                snippet = result.get("description", "")
                content_for_llm = snippet

                if not url or not title:
                    return None

                try:
                    resp = await client.get(url, headers=REQUEST_HEADERS)
                    content_type = resp.headers.get("Content-Type", "").lower()
                    if "text/html" in content_type:
                        soup = BeautifulSoup(resp.text, "lxml")
                        main_elements = soup.find_all(["article", "main"])
                        if main_elements:
                            text = " ".join(
                                get_page_text_content(el) for el in main_elements
                            )
                        elif soup.body:
                            text = get_page_text_content(soup.body)
                        else:
                            text = get_page_text_content(soup)

                        text = text.strip()
                        if len(text) > 5000:
                            print(f"Summarizing long text from {url}")
                            content_for_llm = summarize(text)
                        elif text:
                            content_for_llm = text
                    else:
                        content_for_llm = snippet + " [Note: Non-HTML content]"
                except Exception as e:
                    print(f"Error fetching {url}: {e}")
                    content_for_llm = snippet + " [Note: Scrape failed]"

                return {"title": title, "url": url, "content": content_for_llm.strip()}

            tasks = [fetch_and_process(r) for r in results]
            extracted = await asyncio.gather(*tasks)
            extracted_results = [r for r in extracted if r]

            if r and extracted_results:
                try:
                    r.setex(cache_key, redis_expiry, json.dumps(extracted_results))
                    print("Result cached.")
                except Exception as e:
                    print(f"Failed to cache: {e}")

            return extracted_results

    except Exception as e:
        print(f"Unhandled error during search: {e}")
        return []


# Synchronous wrapper
def brave_web_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    return asyncio.run(_brave_web_search(query, num_results))
