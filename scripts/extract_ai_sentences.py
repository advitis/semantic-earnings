import nltk
from nltk.tokenize import sent_tokenize
from scripts.config import AI_KEYWORDS
import re

nltk.download("punkt")

def _is_substantive(sentence: str) -> bool:
    """Drop greetings, boilerplate, or very short lines."""
    fluff = ("forward-looking", "safe harbor", "good morning", "good afternoon",
             "welcome", "thank you", "operator")
    if len(sentence.split()) < 6:
        return False
    lower = sentence.lower()
    return not any(p in lower for p in fluff)

def _has_ai_keyword(sent: str, keywords=AI_KEYWORDS):
    """
    True if the sentence contains a real AI keyword.
    • 'ai' must appear as a standalone word (word-boundary) or prefix 'ai-'.
    • Other keywords ('machine learning', 'generative ai', …) use substring match.
    """
    txt = sent.lower()
    for kw in keywords:
        if kw == "ai":
            if re.search(r"\bai\b|\bai\-", txt):
                return True
        else:
            if kw in txt:
                return True
    return False

def extract_ai_sentences(text, keywords=AI_KEYWORDS):
    sentences = sent_tokenize(text)
    ai_sentences = [s for s in sentences if _has_ai_keyword(s, keywords)]
    return [s for s in ai_sentences if _is_substantive(s)]
