import pandas as pd
import yaml

from text_scraper import scrape_text_from_url


with open('data_sources.yaml', 'r') as file:
    data_sources = yaml.safe_load(file)

rows_list = [
    {
        'name': key,
        'url': url,
        'text': scrape_text_from_url(url)
    }
    for key, source in data_sources.items()
    for url in source['urls']
]

df = pd.DataFrame(rows_list)
df.to_csv('../data.csv', index=False, escapechar='\\')
