import nltk
from nltk.tokenize import sent_tokenize
from scripts.config import AI_KEYWORDS, BOILERPLATE_PHRASES
import re

nltk.download("punkt")

def _is_substantive(sentence: str) -> bool:
    """
    Returns True if the sentence is substantive (not a greeting, boilerplate, or too short).
    Drops greetings, boilerplate, or very short lines.
    """
    if len(sentence.split()) < 6:
        return False
    lower = sentence.lower()
    return not any(p in lower for p in BOILERPLATE_PHRASES)

def _has_ai_keyword(sent: str, keywords=AI_KEYWORDS):
    """
    Returns True if the sentence contains a real AI keyword.
    - 'ai' must appear as a standalone word (word-boundary) or prefix 'ai-'.
    - Other keywords ('machine learning', 'generative ai', etc.) use substring match.
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
    """
    Extracts substantive sentences from the input text that contain AI-related keywords.
    Filters out non-substantive (boilerplate, greeting, or short) sentences.
    """
    sentences = sent_tokenize(text)
    ai_sentences = [s for s in sentences if _has_ai_keyword(s, keywords)]
    return [s for s in ai_sentences if _is_substantive(s)]
