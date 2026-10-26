"""Flask backend for the Government Scheme Helper.

Routes:
  GET  /               -> the web app
  POST /api/ask        -> answer a question (RAG with category filter)
  POST /api/feedback   -> store thumbs up / down, retrain when a round is due
  GET  /api/progress   -> model version, test scores, retraining history
  GET  /api/categories -> the four supported categories

On first start, missing artefacts (trained model, vector store) are built
automatically so the app is demo-ready.
"""
import sys

from flask import Flask, request, jsonify, send_from_directory

from .config import CATEGORIES, WEB_DIR, MODEL_PATH, VECTOR_STORE_PATH
from .rag import answer_question
from . import feedback as fb

app = Flask(__name__, static_folder=None)


def _ensure_artifacts() -> None:
    if not MODEL_PATH.exists():
        print("No trained model found — training the classifier now (first run only)")
        from .classifier import train_full
        train_full()
    if not VECTOR_STORE_PATH.exists():
        print("No vector store found — building it now (first run only)")
        from .build_vector_store import build_vector_store
        build_vector_store()


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/api/categories")
def categories():
    return jsonify({"categories": CATEGORIES})


@app.post("/api/ask")
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Please type a question."}), 400
    try:
        return jsonify(answer_question(question))
    except FileNotFoundError as exc:
        return jsonify({"error": f"Model artefacts missing: {exc}. "
                                 f"Run the build scripts in README first."}), 500


@app.post("/api/feedback")
def give_feedback():
    data = request.get_json(silent=True) or {}
    try:
        status = fb.record_feedback(
            question=data.get("question", ""),
            predicted_category=data.get("predicted_category", ""),
            label=data.get("label", ""),
            correct_category=data.get("correct_category"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    result = fb.maybe_retrain()
    return jsonify({**status, "retrained": result["retrained"],
                    "round": result.get("round")})


@app.get("/api/progress")
def progress():
    return jsonify(fb.progress_report())


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    _ensure_artifacts()
    import os
    port = int(os.getenv("PORT", "8000"))
    print(f"Government Scheme Helper running at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
