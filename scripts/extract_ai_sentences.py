import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

AI_KEYWORDS = [
    "artificial intelligence", "ai", "machine learning", "ml", "neural network",
    "deep learning", "large language model", "llm", "gpt", "automation"
]


def extract_ai_sentences(text, keywords=AI_KEYWORDS):
    sentences = sent_tokenize(text)
    return [s for s in sentences if any(k in s.lower() for k in keywords)]
