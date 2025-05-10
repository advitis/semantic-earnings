import os
import requests
from dotenv import load_dotenv
load_dotenv()

CACHE_DIR = "data/transcripts"
API_KEY = os.getenv("API_NINJAS_KEY")
BASE_URL = "https://api.api-ninjas.com/v1/earningstranscript"


def fetch_transcript(ticker, year, quarter):
    os.makedirs(CACHE_DIR, exist_ok=True)
    filename = f"{CACHE_DIR}/{ticker}_{year}_Q{quarter}.txt"
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()

    print(f"[API] Fetching {ticker} {year} Q{quarter}...")
    headers = {"X-Api-Key": API_KEY}
    params = {"ticker": ticker, "year": year, "quarter": quarter}
    r = requests.get(BASE_URL, headers=headers, params=params)
    if r.status_code == 200:
        text = r.json().get("transcript", "")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        return text
    else:
        print(f"[ERROR] Failed to fetch {ticker} {year} Q{quarter} - {r.status_code}")
        return ""
