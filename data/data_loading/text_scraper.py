import logging
import re
from io import BytesIO

import PyPDF2
import requests
from trafilatura import fetch_url, extract


def scrape_text_from_url(url):
    if url is None:
        return ""

    scraping_functions = {
        "html": scrape_html,
        "pdf": scrape_pdf,
    }
    content_type = fetch_content_type(url)
    if content_type in scraping_functions:
        logging.warning(f"Scraped text from {url}...")
        return scraping_functions[content_type](url)
    else:
        raise ValueError(f"Unsupported Content Type: {content_type}")


def fetch_content_type(url):
    try:
        response = requests.get(url)
        content_type = response.headers.get('Content-Type')
        content_type_extracted = re.search(r'(pdf|html)', content_type).group(1)
        return content_type_extracted
    except Exception as e:
        raise ValueError(f'Error fetching content type for {url}: {str(e)}')


def scrape_html(url):
    downloaded = fetch_url(url)
    return extract(downloaded)


def scrape_pdf(url):
    response = requests.get(url)

    with BytesIO(response.content) as data:
        pdf = PyPDF2.PdfReader(data)

        result = ''.join(page.extract_text() for page in pdf.pages)
        return result
