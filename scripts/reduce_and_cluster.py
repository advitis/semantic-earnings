from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP
import re


def reduce_embeddings(embeddings, n_neighbors=5, min_dist=0.3, metric="cosine", random_state=42):
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state)
    return reducer.fit_transform(embeddings)


def cluster_embeddings(embeddings, k=10, random_state=0):
    """K‑means wrapper returning one label per embedding."""
    return KMeans(n_clusters=k, n_init="auto", random_state=random_state).fit_predict(embeddings)


def _clean_term(term: str) -> bool:
    """Return True if the candidate term is meaningful (no digits, not stop‑ filler)."""
    bad_tokens = {
        "risk", "risks", "uncertainties", "forward", "looking", "statement", "statements",
        "safe", "harbor", "thank", "thanks", "welcome", "operator",
        "quarter", "quarters", "year", "years", "think", "said", "business", "question", "subject", "remain", "currency",
        "growth", "focused", "actual", "results", "differ", "cash", "revenue", "expect", "nvidia", "income",
        "learning", "computing", "seamless", "seamlessly", "just", "going", "continue", "meta", "world", "really",
        "driven", "people", "time", "margin", "new", "term"
    }
    if any(tok in bad_tokens for tok in term.split()):
        return False
    if re.search(r"\d", term):
        return False
    if len(term) < 3:
        return False
    return True


def label_clusters(sentences, labels, top_n=3):
    """
    Return {cluster_id: 'keyword1 / keyword2'} using mean TF‑IDF,
    filtering out numeric junk and boilerplate filler words.
    """
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english")
    X = vectorizer.fit_transform(sentences)
    terms = vectorizer.get_feature_names_out()
    names = {}
    for c in set(labels):
        idx = [i for i, l in enumerate(labels) if l == c]
        if not idx:
            continue
        mean_scores = X[idx].mean(axis=0).A1
        ordered = mean_scores.argsort()[::-1]
        picked = []
        for j in ordered:
            candidate = terms[j]
            if _clean_term(candidate):
                picked.append(candidate)
            if len(picked) == top_n:
                break
        names[c] = " / ".join(picked) if picked else "misc"
    return names
