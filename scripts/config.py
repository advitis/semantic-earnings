COMPANIES = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "meta": "META",
    "amazon": "AMZN",
    "nvidia": "NVDA",
    "ibm": "IBM",
    "oracle": "ORCL"
}

AI_KEYWORDS = [
    "artificial intelligence", "ai", "machine learning", "ml", "neural network",
    "deep learning", "large language model", "llm", "gpt", "automation"
]

BOILERPLATE_PHRASES = (
        "forward-looking", "safe harbor", "good morning", "good afternoon",
        "welcome", "thank you", "operator"
    )

BAD_TOKENS = {
        "risk", "risks", "uncertainties", "forward", "looking", "statement", "statements",
        "safe", "harbor", "thank", "thanks", "welcome", "operator",
        "quarter", "quarters", "year", "years", "think", "said", "business", "question", "subject", "remain", "currency",
        "growth", "focused", "actual", "results", "differ", "cash", "revenue", "expect", "nvidia", "income",
        "learning", "computing", "seamless", "seamlessly", "just", "going", "continue", "meta", "world", "really",
        "driven", "people", "time", "margin", "new", "term"
    }

YEARS = range(2019, 2025)
QUARTERS = [1, 2, 3, 4]

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_CACHE = "embeddings/ai_embeddings.pkl"

TRANSCRIPTS_PATH = "data/transcripts"
