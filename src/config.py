"""Shared paths and constants for the Government Scheme Helper project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
STORAGE_DIR = PROJECT_ROOT / "storage"
WEB_DIR = PROJECT_ROOT / "web"

SCHEMES_CSV = DATA_DIR / "schemes.csv"
FLASHCARDS_CSV = DATA_DIR / "flashcards.csv"
EVALUATION_CSV = DATA_DIR / "evaluation_set.csv"

FEEDBACK_CSV = STORAGE_DIR / "feedback.csv"
PROGRESS_JSON = STORAGE_DIR / "progress.json"

MODEL_PATH = MODELS_DIR / "current_model.joblib"
VERSIONS_DIR = MODELS_DIR / "versions"
VECTOR_STORE_PATH = MODELS_DIR / "vector_store.joblib"
EMBEDDER_PATH = MODELS_DIR / "embedder.joblib"

CATEGORIES = ["Farmer", "Student", "Woman and Child", "Senior Citizen"]

# Retrieval returns this many chunks (the doc's "top 3 chunks").
TOP_K = 3

# A retraining round starts after this many new labelled feedback examples.
RETRAIN_THRESHOLD = 50

RANDOM_STATE = 42

# Similarity floor per embedding backend, below which the app says
# "no relevant scheme found" instead of showing a weak match.
MIN_SIMILARITY = {
    "fastembed": 0.35,
    "st": 0.25,
    "tfidf": 0.05,
}
