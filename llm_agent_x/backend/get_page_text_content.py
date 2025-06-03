from bs4 import BeautifulSoup


def get_page_text_content(soup: BeautifulSoup) -> str:
    """
    Extracts text content from a BeautifulSoup object.
    Tries to be a bit smarter than just soup.get_text() by removing script/style.
    Still, this is a heuristic and might not be perfect for all pages.
    """
    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    # Get text, try to preserve some structure with spaces
    text = soup.get_text(separator=" ", strip=True)

    # Optional: Further clean-up (e.g., multiple newlines, excessive whitespace)
    # text = "\n".join([line for line in text.splitlines() if line.strip()])
    # text = re.sub(r'\s+', ' ', text).strip()
    return text
