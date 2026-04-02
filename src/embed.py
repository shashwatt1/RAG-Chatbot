import numpy as np
from sentence_transformers import SentenceTransformer

# load once at import time
_model = SentenceTransformer('all-MiniLM-L6-v2')


def get_dimension():
    return _model.get_sentence_embedding_dimension()


def get_embeddings(texts):
    if not texts:
        return np.array([])
    vecs = _model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # L2-normalize so we can use cosine similarity with IndexFlatIP
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid div by zero
    return vecs / norms
