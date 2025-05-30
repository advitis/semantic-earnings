# How Big Tech is Talking About AI: Earnings Call Clustering (2019–2025)

Please refer to my [Medium](https://medium.com/@advitis/how-big-tech-is-talking-about-ai-what-earnings-calls-reveal-2019-2025-33ec4b08b289) article for a narrative deep-dive of this analysis.

This project analyzes 200+ quarterly earnings calls from major tech companies to uncover how they talk about **Artificial Intelligence** - not through analyst summaries, but using **Natural Language Processing (NLP)** to map their AI narratives directly.

Each dot in the final chart represents an earnings call. Similar calls cluster together based on what was actually said - revealing strategic pivots, converging agendas, and the emergence of generative AI.

![Semantic Map Screenshot](./semantic_map.png)

---

## 🧠 What This Project Does

### → Goal
Turn six years of text-heavy earnings transcripts into a **semantic map** of AI narratives.

### → Method
1. **Filter** sentences for true AI content (not fluff or legal disclaimers)
2. **Embed** using `sentence-transformers` (MiniLM)
3. **Cluster** with KMeans to detect recurring themes
4. **Name clusters** using cleaned TF-IDF keyword ranking
5. **Aggregate** sentence themes to the earnings-call level
6. **Visualize** with UMAP (dimensionality reduction) + Plotly

---

## 📦 Key Files

| File | Purpose |
|------|---------|
| `scripts/extract_ai_sentences.py` | Filters meaningful AI mentions |
| `scripts/embed_sentences.py` | Runs `SentenceTransformer` encoding |
| `scripts/reduce_and_cluster.py` | UMAP projection + KMeans clustering + cluster labelling |
| `scripts/visualize.py` | Generates final semantic map (with hover + zoom options) |
| `main.py` | Pulls it all together into one pipeline |

---

## 🗺️ Strategic Insight (See Medium article for more detail)

While this repo focuses on implementation, the visual output reveals major trends:

- **Meta** sustains a long-term focus on ad optimization, with a parallel shift from 2023 onward toward generative AI initiatives such as virtual assistants.
- **Amazon** and **IBM** increasingly converge in narrative by 2023–24, emphasizing enterprise-grade generative AI platforms (e.g., Bedrock, Watsonx).
- **Apple** evolves from emphasizing on-device personalization to reintroducing infrastructure themes, notably through its “Private Compute Cloud” strategy.

---
