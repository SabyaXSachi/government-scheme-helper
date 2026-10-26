"""Vector store for the retrieval layer.

For a corpus of a few hundred chunks, exact cosine similarity search is the
same result a flat FAISS index (IndexFlatIP on normalised vectors) would
return, at negligible cost. If FAISS happens to be installed the search
delegates to it; otherwise a plain numpy implementation is used. The
interface (add / search with a metadata category filter / save / load) is
what a Chroma or FAISS collection would expose for this use case.
"""
import numpy as np
import joblib

try:  # optional
    import faiss
    _FAISS = True
except ImportError:
    _FAISS = False


class VectorStore:
    def __init__(self):
        self.texts: list[str] = []
        self.metadatas: list[dict] = []
        self.embeddings: np.ndarray | None = None  # (n, dim), L2-normalised

    def add(self, texts, metadatas, embeddings: np.ndarray) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

    def _subset(self, category: str | None) -> list[int]:
        if category is None:
            return list(range(len(self.texts)))
        return [i for i, m in enumerate(self.metadatas)
                if m.get("category") == category]

    def search(self, query_vec: np.ndarray, k: int = 3,
               category: str | None = None) -> list[dict]:
        """Return the k closest chunks (optionally only within one category),
        as [{text, metadata, similarity}] sorted by similarity."""
        indices = self._subset(category)
        if not indices:
            return []
        query_vec = np.asarray(query_vec, dtype=np.float32).ravel()
        subset = self.embeddings[indices]

        if _FAISS:
            index = faiss.IndexFlatIP(subset.shape[1])
            index.add(np.ascontiguousarray(subset))
            scores, local = index.search(query_vec.reshape(1, -1), min(k, len(indices)))
            top = list(zip(local[0].tolist(), scores[0].tolist()))
        else:
            scores = subset @ query_vec
            order = np.argsort(-scores)[: min(k, len(indices))]
            top = [(int(i), float(scores[i])) for i in order]

        hits = []
        for local_idx, sim in top:
            if sim <= 0.0:
                continue
            i = indices[local_idx]
            hits.append({"text": self.texts[i],
                         "metadata": self.metadatas[i],
                         "similarity": round(float(sim), 4)})
        return hits

    def save(self, path) -> None:
        joblib.dump({"texts": self.texts, "metadatas": self.metadatas,
                     "embeddings": self.embeddings}, path)

    @classmethod
    def load(cls, path) -> "VectorStore":
        data = joblib.load(path)
        store = cls()
        store.texts = data["texts"]
        store.metadatas = data["metadatas"]
        store.embeddings = data["embeddings"]
        return store
