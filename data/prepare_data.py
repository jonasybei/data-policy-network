import re
import string
import pandas as pd


def tokenize_text(text):
    if pd.isna(text):
        return []
    if not isinstance(text, str):
        print(f"There has been one error.")
        return []

    return re.sub('[' + string.punctuation + ']', '', text).split()


df = pd.read_csv("data.csv")
df['tokenized_text'] = df['text'].apply(tokenize_text)

df.to_csv("data.csv", index=False)
