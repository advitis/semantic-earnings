import pandas as pd
import streamlit as st
from scripts.visualize import plot_semantic_map
from scripts.config import COMPANIES, YEARS


st.title("🧠 Semantic AI Tracker")
st.markdown("Analyze how companies talk about AI in earnings calls over time.")

selected_companies = st.multiselect("Select companies", list(COMPANIES.keys()), default=list(COMPANIES.keys()))
selected_years = st.slider("Select year range", min_value=min(YEARS), max_value=max(YEARS), value=(2020, 2024))

if st.button("Show Map"):
    # Load pre‑computed call‑level dataframe
    call_df = pd.read_pickle("processed/call_df.pkl")

    # Apply UI filters
    mask = (
        call_df["company"].isin(selected_companies)
        & call_df["year"].between(*selected_years)
    )
    filtered = call_df[mask]

    if filtered.empty:
        st.warning("No calls match the selected filters.")
        st.stop()

    st.success("Loaded pre‑computed embeddings and coordinates.")
    plot_semantic_map(filtered)
