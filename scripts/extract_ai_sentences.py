import nltk
from nltk.tokenize import sent_tokenize
from config import AI_KEYWORDS

nltk.download("punkt")

def extract_ai_sentences(text, keywords=AI_KEYWORDS):
    sentences = sent_tokenize(text)
    return [s for s in sentences if any(k in s.lower() for k in keywords)]
