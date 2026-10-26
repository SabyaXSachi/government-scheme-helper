"""RAG pipeline: predict category -> retrieve within it -> grounded answer.

Flow of one question (architecture stages 3):
  clean text -> TF-IDF -> classifier predicts category
  embed question -> vector search only within that category -> top 3 chunks
  LLM (if configured) or extractive generator writes the answer from ONLY
  the retrieved text, with official source links; says so if nothing found.
"""
import os

import pandas as pd

from .config import (SCHEMES_CSV, MODEL_PATH, VECTOR_STORE_PATH,
                     EMBEDDER_PATH, TOP_K)
from .text_utils import clean_text
from .classifier import predict_category
from .embeddings import load_embedder
from .vector_store import VectorStore

INFO_NOTE = ("This is general information from official government sources "
             "and does not confirm eligibility.")

_runtime = None


def _load_scheme_records() -> dict[str, dict]:
    df = pd.read_csv(SCHEMES_CSV).fillna("")
    return {row["scheme_name"]: row.to_dict() for _, row in df.iterrows()}


def _load_pickle(path):
    import joblib
    return joblib.load(path)


def load_runtime() -> dict:
    """Load all artefacts needed to answer questions (cached)."""
    global _runtime
    if _runtime is not None and _runtime.get("bundle") is not None:
        return _runtime
    _runtime = {
        "bundle": _load_pickle(MODEL_PATH),
        "store": VectorStore.load(VECTOR_STORE_PATH),
        "embedder": load_embedder(EMBEDDER_PATH),
        "records": _load_scheme_records(),
    }
    return _runtime


def _llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _llm_answer(question: str, hits: list[dict]) -> str | None:
    """Ask the configured OpenAI-compatible API to write the answer strictly
    from the retrieved chunks. Returns None on any failure so the caller can
    fall back to the extractive answer."""
    import requests
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    context = "\n\n".join(
        f"[{i + 1}] {h['metadata']['scheme_name']} (source: {h['metadata']['source_url']})\n"
        f"{h['text']}" for i, h in enumerate(hits))
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            json={
                "model": model,
                "temperature": 0.1,
                "messages": [
                    {"role": "system",
                     "content": ("You answer questions about Indian government welfare "
                                 "schemes using ONLY the numbered scheme text provided. "
                                 "Name the schemes you used. If the provided text does "
                                 "not contain the answer, say so plainly. Be concise.")},
                    {"role": "user",
                     "content": f"Question: {question}\n\nScheme text:\n{context}"},
                ],
            },
            timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def _extractive_answer(hits: list[dict]) -> str:
    names = " and ".join(h["metadata"]["scheme_name"] for h in hits[:2])
    if len(hits) == 1:
        return f"Based on official scheme information, the closest match is {names}."
    return (f"Based on official scheme information, {len(hits)} schemes look "
            f"relevant. The closest match is {names}.")


def answer_question(question: str) -> dict:
    rt = load_runtime()
    bundle, store = rt["bundle"], rt["store"]
    embedder, records = rt["embedder"], rt["records"]

    cleaned = clean_text(question)
    category, confidence = predict_category(bundle, question)
    qvec = embedder.encode([cleaned])[0]

    hits = store.search(qvec, k=TOP_K, category=category)
    min_sim = embedder.min_similarity
    if not hits or hits[0]["similarity"] < min_sim:
        return {
            "question": question,
            "predicted_category": category,
            "confidence": confidence,
            "found": False,
            "answer": ("No relevant scheme was found in the official corpus for "
                       "this category. Try describing your situation differently."),
            "schemes": [],
            "sources": [],
            "note": INFO_NOTE,
            "model": _model_info(rt),
        }

    # Deduplicate chunks by scheme, keeping retrieval order.
    scheme_names: list[str] = []
    for hit in hits:
        name = hit["metadata"]["scheme_name"]
        if name not in scheme_names:
            scheme_names.append(name)

    schemes = []
    for name in scheme_names:
        rec = records[name]
        best_sim = max(h["similarity"] for h in hits if h["metadata"]["scheme_name"] == name)
        schemes.append({
            "scheme_name": rec["scheme_name"],
            "category": rec["category"],
            "eligibility": rec["eligibility"],
            "benefits": rec["benefits"],
            "how_to_apply": rec["how_to_apply"],
            "documents_required": rec["documents_required"],
            "source_url": rec["source_url"],
            "similarity": best_sim,
        })

    answer = _llm_answer(question, hits) if _llm_available() else None
    if answer is None:
        answer = _extractive_answer(hits)

    return {
        "question": question,
        "predicted_category": category,
        "confidence": confidence,
        "found": True,
        "answer": answer,
        "schemes": schemes,
        "sources": [s["source_url"] for s in schemes],
        "note": INFO_NOTE,
        "model": _model_info(rt),
    }


def _model_info(rt: dict) -> dict:
    bundle = rt["bundle"]
    return {
        "classifier": bundle["kind"],
        "classifier_version": bundle["version"],
        "embedding_backend": rt["embedder"].backend,
        "top_k": TOP_K,
        "category_filter": True,
    }
