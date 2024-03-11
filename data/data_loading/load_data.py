import pandas as pd
from text_scraper import scrape_text_from_url

data_sources_path = 'data_sources.csv'
df = pd.read_csv(data_sources_path)
df['text'] = df['url'].apply(scrape_text_from_url)

df.to_csv('../data.csv', index=False, escapechar='"')
