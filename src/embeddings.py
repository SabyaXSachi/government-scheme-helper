"""Embedding backends for the retrieval layer.

The project document calls for a pre-trained sentence embedding model.
This module tries, in order:

1. fastembed (ONNX, small download, no PyTorch)
2. sentence-transformers (the classic Sentence-BERT stack)
3. a TF-IDF based fallback so the app runs with zero extra downloads

The chosen backend is persisted with the vector store so queries and
documents always use the same embedding space.
"""
import os

import numpy as np
import joblib

from .config import MIN_SIMILARITY


class TfidfEmbedder:
    """Lexical fallback: TF-IDF vectors (single words + word pairs) used as
    embeddings. Not semantic, but a reasonable stand-in for a small corpus."""

    def __init__(self):
        self.backend = "tfidf"
        self.model_name = "tfidf-1-2gram"
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                           stop_words="english")

    @property
    def min_similarity(self) -> float:
        return MIN_SIMILARITY["tfidf"]

    @property
    def dim(self) -> int:
        return len(self._vectorizer.vocabulary_)

    def fit(self, texts):
        self._vectorizer.fit(texts)
        return self

    def encode(self, texts) -> np.ndarray:
        matrix = self._vectorizer.transform(texts).astype(np.float32).toarray()
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def state(self) -> dict:
        return {"backend": self.backend, "model_name": self.model_name,
                "vectorizer": self._vectorizer}


class FastEmbedEmbedder:
    """Pre-trained sentence embeddings via fastembed (ONNX)."""

    MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self):
        from fastembed import TextEmbedding
        self.backend = "fastembed"
        self.model_name = self.MODEL
        self._model = TextEmbedding(model_name=self.MODEL)

    @property
    def min_similarity(self) -> float:
        return MIN_SIMILARITY["fastembed"]

    @property
    def dim(self) -> int:
        return 384

    def fit(self, texts):
        return self  # pre-trained; nothing to fit

    def encode(self, texts) -> np.ndarray:
        vectors = np.array(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def state(self) -> dict:
        return {"backend": self.backend, "model_name": self.model_name}


class STEmbedder:
    """Pre-trained sentence embeddings via sentence-transformers."""

    MODEL = "all-MiniLM-L6-v2"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.backend = "st"
        self.model_name = self.MODEL
        self._model = SentenceTransformer(self.MODEL)

    @property
    def min_similarity(self) -> float:
        return MIN_SIMILARITY["st"]

    @property
    def dim(self) -> int:
        return 384

    def fit(self, texts):
        return self

    def encode(self, texts) -> np.ndarray:
        vectors = np.asarray(self._model.encode(texts, normalize_embeddings=True),
                             dtype=np.float32)
        return vectors

    def state(self) -> dict:
        return {"backend": self.backend, "model_name": self.model_name}


def get_embedder(backend: str | None = None):
    """Create an embedder. Order: explicit arg > env var > auto-detect."""
    backend = backend or os.getenv("EMBEDDING_BACKEND")
    if backend:
        backend = backend.lower()
        if backend == "tfidf":
            return TfidfEmbedder()
        if backend == "fastembed":
            return FastEmbedEmbedder()
        if backend == "st":
            return STEmbedder()
        raise ValueError(f"Unknown embedding backend: {backend}")

    for cls in (FastEmbedEmbedder, STEmbedder):
        try:
            return cls()
        except ImportError:
            continue
    return TfidfEmbedder()


def save_embedder(embedder, path) -> None:
    joblib.dump(embedder.state(), path)


def load_embedder(path):
    state = joblib.load(path)
    embedder = get_embedder(state["backend"])
    if isinstance(embedder, TfidfEmbedder):
        embedder._vectorizer = state["vectorizer"]
    return embedder
