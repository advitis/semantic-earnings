from sentence_transformers import SentenceTransformer
import os
import pickle
from scripts.config import EMBED_CACHE, EMBED_MODEL

model = SentenceTransformer(EMBED_MODEL)


def compute_embeddings(texts, cache_path=EMBED_CACHE, use_cache=True):
    if use_cache and os.path.exists(cache_path):
        cached = pickle.load(open(cache_path, "rb"))
        if len(cached) == len(texts):
            print("✅ Using cached embeddings.")
            return cached
        else:
            print("⚠️ Cache size mismatch — regenerating embeddings.")

    embeddings = model.encode(texts, show_progress_bar=True)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings
