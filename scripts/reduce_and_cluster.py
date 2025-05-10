import numpy as np
from umap import UMAP


def reduce_embeddings(embeddings, n_neighbors=5, min_dist=0.3, metric="cosine"):
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    return reducer.fit_transform(embeddings)
