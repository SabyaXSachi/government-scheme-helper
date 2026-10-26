"""CLI: build the vector store from the scheme corpus (chunk -> embed -> store)."""
import sys

import pandas as pd

from .config import SCHEMES_CSV, VECTOR_STORE_PATH, EMBEDDER_PATH
from .text_utils import chunk_text, scheme_text
from .embeddings import get_embedder, save_embedder
from .vector_store import VectorStore


def build_vector_store() -> VectorStore:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_csv(SCHEMES_CSV).fillna("")
    embedder = get_embedder()
    store = VectorStore()

    texts, metas = [], []
    for _, row in df.iterrows():
        record = row.to_dict()
        full_text = scheme_text(record)
        for chunk in chunk_text(full_text):
            texts.append(chunk)
            metas.append({"scheme_name": record["scheme_name"],
                          "category": record["category"],
                          "source_url": record["source_url"]})

    print(f"Embedding backend: {embedder.backend} ({embedder.model_name})")
    print(f"{len(df)} schemes -> {len(texts)} chunks")
    embedder.fit(texts)
    embeddings = embedder.encode(texts)
    store.add(texts, metas, embeddings)

    store.save(VECTOR_STORE_PATH)
    save_embedder(embedder, EMBEDDER_PATH)
    print(f"Vector store saved to {VECTOR_STORE_PATH}")
    return store


if __name__ == "__main__":
    build_vector_store()
