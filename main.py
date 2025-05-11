import pandas as pd
from pathlib import Path
from scripts.extract_ai_sentences import extract_ai_sentences
from scripts.embed_sentences import compute_embeddings
from scripts.reduce_and_cluster import reduce_embeddings
from scripts.visualize import plot_semantic_map
from scripts.config import COMPANIES, TRANSCRIPTS_PATH

def collect_ai_data():
    rows = []
    for file in Path(TRANSCRIPTS_PATH).glob("*.txt"):
        ticker, year, quarter = file.stem.split("_")
        company = next((k for k, v in COMPANIES.items() if v == ticker), "unknown")
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            continue
        mentions = extract_ai_sentences(text)
        if not mentions:
            continue
        rows.append({
            "company": company,
            "ticker": ticker,
            "year": int(year),
            "quarter": int(quarter.replace("Q", "")),
            "text": " ".join(mentions),
            "mention_count": len(mentions)
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = collect_ai_data()
    df = df[df["text"].str.strip() != ""].copy()
    embeddings = compute_embeddings(df["text"].tolist())
    coords = reduce_embeddings(embeddings)

    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    plot_semantic_map(df)
