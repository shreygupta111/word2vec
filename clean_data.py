import nltk
from nltk.corpus import stopwords

import re
nltk.download('stopwords')

stopwords_list = stopwords.words("english")
def change_lower(text):
    return text.lower()

def clean_corpus(text):
    text = re.sub(r'[^ \nA-Za-z0-9À-ÖØ-öø-ÿ/]+', '', text)
    text = re.sub(r'[\\/×\^\]\[÷]', '', text)
    return text

def remove_stopwords(text: str):
    text_tokens = text.split(" ")
    text_tokens = [tokens for tokens in text_tokens if not tokens in stopwords_list]
    return " ".join(text_tokens)

def preprocess(df, column_name):
    return df[column_name].astype(str).apply(change_lower).apply(clean_corpus).apply(remove_stopwords)
