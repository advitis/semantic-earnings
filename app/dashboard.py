from pathlib import Path
import streamlit as st
import pandas as pd
from scripts.visualize import plot_semantic_map
from scripts.embed_sentences import compute_embeddings
from scripts.reduce_and_cluster import reduce_embeddings
from scripts.extract_ai_sentences import extract_ai_sentences
from scripts.config import COMPANIES, YEARS, TRANSCRIPTS_PATH


st.title("🧠 Semantic AI Tracker")
st.markdown("Analyze how companies talk about AI in earnings calls over time.")

selected_companies = st.multiselect("Select companies", list(COMPANIES.keys()), default=list(COMPANIES.keys()))
selected_years = st.slider("Select year range", min_value=min(YEARS), max_value=max(YEARS), value=(2020, 2024))

if st.button("Run Analysis"):
    rows = []
    with st.spinner("Loading local transcripts..."):
        for file in Path(TRANSCRIPTS_PATH).glob("*.txt"):
            ticker, year, quarter = file.stem.split("_")
            company = next((k for k, v in COMPANIES.items() if v == ticker), "unknown")
            if company not in selected_companies:
                continue
            year = int(year)
            if year < selected_years[0] or year > selected_years[1]:
                continue
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
                "year": year,
                "quarter": int(quarter.replace("Q", "")),
                "text": " ".join(mentions),
                "mention_count": len(mentions)
            })

    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("No AI-related mentions found for selected filters.")
    else:
        embeddings = compute_embeddings(df["text"].tolist(), use_cache=False)
        coords = reduce_embeddings(embeddings)
        df["x"] = coords[:, 0]
        df["y"] = coords[:, 1]
        st.success("Visualization ready!")
        plot_semantic_map(df)
