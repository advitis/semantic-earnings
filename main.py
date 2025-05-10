import pandas as pd

from scripts.fetch_transcripts import fetch_transcript
from scripts.extract_ai_sentences import extract_ai_sentences
from scripts.embed_sentences import compute_embeddings
from scripts.reduce_and_cluster import reduce_embeddings
from scripts.visualize import plot_semantic_map

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
YEARS = [2023, 2024]
QUARTERS = [1, 2, 3, 4]


def collect_ai_data():
    rows = []
    for company, ticker in COMPANIES.items():
        for year in YEARS:
            for q in QUARTERS:
                text = fetch_transcript(ticker, year, q)
                if not text.strip():
                    continue
                mentions = extract_ai_sentences(text)
                if not mentions:
                    continue
                joined = " ".join(mentions)
                rows.append({
                    "company": company,
                    "ticker": ticker,
                    "year": year,
                    "quarter": q,
                    "text": joined,
                    "mention_count": len(mentions)
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = collect_ai_data()
    embeddings = compute_embeddings(df["text"].tolist())
    coords = reduce_embeddings(embeddings)

    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    plot_semantic_map(df)
