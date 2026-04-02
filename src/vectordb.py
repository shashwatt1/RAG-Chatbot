import os
import json
import faiss
import numpy as np


class VectorDB:
    def __init__(self, index_path="index.faiss", meta_path="metadata.json", dimension=384):
        self.index_path = index_path
        self.meta_path = meta_path
        self.dimension = dimension
        self.index = None
        self.metadata = []

    def load(self):
        # restore index + metadata from disk if available
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            return True
        return False

    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f)

    def build_or_update(self, embeddings, chunks_meta):
        if len(embeddings) == 0:
            return

        if self.index is None:
            # IndexFlatIP + normalized vecs = cosine similarity
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []

        self.index.add(np.array(embeddings, dtype=np.float32))
        self.metadata.extend(chunks_meta)
        self.save()

    def search(self, query_vec, top_k=5):
        if self.index is None or self.index.ntotal == 0:
            return []

        # scores here are cosine similarity (higher = more relevant)
        scores, indices = self.index.search(np.array(query_vec, dtype=np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata) and float(score) >= 0.1:
                results.append((self.metadata[idx], float(score)))

        return results
