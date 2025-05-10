import streamlit as st
import pandas as pd
from scripts.visualize import plot_semantic_map
from scripts.embed_sentences import compute_embeddings
from scripts.reduce_and_cluster import reduce_embeddings
from scripts.extract_ai_sentences import extract_ai_sentences
from scripts.fetch_transcripts import fetch_transcript

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

st.title("🧠 Semantic AI Tracker")
st.markdown("Analyze how companies talk about AI in earnings calls over time.")

selected_companies = st.multiselect("Select companies", list(COMPANIES.keys()), default=list(COMPANIES.keys()))
selected_years = st.slider("Select year range", min_value=min(YEARS), max_value=max(YEARS), value=(2020, 2024))

if st.button("Run Analysis"):
    rows = []
    with st.spinner("Fetching and processing transcripts..."):
        for company in selected_companies:
            ticker = COMPANIES[company]
            for year in range(*selected_years):
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
