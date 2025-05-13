import pandas as pd
import numpy as np
from pathlib import Path
from scripts.extract_ai_sentences import extract_ai_sentences
from scripts.embed_sentences import compute_embeddings
from scripts.reduce_and_cluster import reduce_embeddings, cluster_embeddings, label_clusters
from scripts.visualize import plot_semantic_map
from scripts.config import COMPANIES, TRANSCRIPTS_PATH

def collect_ai_sentences():
    """
    Collects all sentences from transcript files that are substantive and contain AI keywords.
    Returns a DataFrame with columns: company, call_id, year, quarter, sentence.
    """
    rows = []
    for file in Path(TRANSCRIPTS_PATH).glob("*.txt"):
        ticker, year, quarter = file.stem.split("_")
        company = next((k for k, v in COMPANIES.items() if v == ticker), "unknown")
        call_id = f"{company}_{year}_Q{quarter.replace('Q','')}"
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            continue
        sentences = extract_ai_sentences(text)
        if not sentences:
            continue
        for s in sentences:
            rows.append({
                "company": company,
                "call_id": call_id,
                "year": int(year),
                "quarter": int(quarter.replace("Q", "")),
                "sentence": s.strip()
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    # Sentence‑level dataframe
    df_sent = collect_ai_sentences()
    df_sent = df_sent[df_sent["sentence"].str.strip() != ""].copy()

    # Sentence embeddings
    sent_embeddings = compute_embeddings(df_sent["sentence"].tolist())

    # Sentence clusters + names
    sent_labels = cluster_embeddings(sent_embeddings, k=15)
    df_sent["cluster"] = sent_labels
    cluster_names = label_clusters(df_sent["sentence"].tolist(), sent_labels, top_n=2)
    df_sent["cluster_name"] = df_sent["cluster"].map(cluster_names)

    # Aggregate to call‑level
    call_rows = []
    call_embeddings = []
    for call_id, grp in df_sent.groupby("call_id"):
        call_rows.append({
            "call_id": call_id,
            "company": grp["company"].iloc[0],
            "year": grp["year"].iloc[0],
            "quarter": grp["quarter"].iloc[0],
            "sentence_count": len(grp),
            # dominant theme of the call
            "cluster_name": grp["cluster_name"].value_counts().idxmax()
        })
        call_embeddings.append(sent_embeddings[grp.index].mean(axis=0))

    call_df = pd.DataFrame(call_rows)

    # 2‑D projection of call embeddings
    coords = reduce_embeddings(np.vstack(call_embeddings))
    call_df["x"], call_df["y"] = coords[:, 0], coords[:, 1]

    # Visual
    plot_semantic_map(call_df)
