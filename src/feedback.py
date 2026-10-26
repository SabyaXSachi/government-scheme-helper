"""Feedback loop and model improvement (architecture stage 4).

- thumbs up  -> question + predicted category becomes a labelled example
- thumbs down-> the user's chosen category becomes the label
- feedback examples go only into training data, never the locked test set
- retraining happens in rounds (~RETRAIN_THRESHOLD new examples)
- the new model replaces the current one only if its test score is not worse;
  every version is kept and the score history drives the progress page
"""
import csv
import json
from datetime import datetime

import pandas as pd

from .config import (CATEGORIES, FEEDBACK_CSV, FLASHCARDS_CSV, MODEL_PATH,
                     PROGRESS_JSON, RETRAIN_THRESHOLD, STORAGE_DIR,
                     VERSIONS_DIR, RANDOM_STATE)
from .classifier import (load_bundle, load_flashcards, stratified_split,
                         _refit, evaluate_on_test)
from .text_utils import clean_text

_FEEDBACK_FIELDS = ["timestamp", "question", "predicted_category",
                    "correct_category", "label"]


def _append_csv(path, fields, row) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def record_feedback(question: str, predicted_category: str, label: str,
                    correct_category: str | None = None) -> dict:
    if label not in ("up", "down"):
        raise ValueError("label must be 'up' or 'down'")
    if label == "down" and correct_category not in CATEGORIES:
        raise ValueError("a thumbs down needs the correct category")
    _append_csv(FEEDBACK_CSV, _FEEDBACK_FIELDS, {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question.strip(),
        "predicted_category": predicted_category,
        "correct_category": correct_category or "",
        "label": label,
    })
    return pending_status()


def _load_progress() -> dict:
    if PROGRESS_JSON.exists():
        return json.loads(PROGRESS_JSON.read_text(encoding="utf-8"))
    return {"rounds": [], "consumed_feedback_rows": 0}


def _save_progress(progress: dict) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_JSON.write_text(json.dumps(progress, indent=2, ensure_ascii=False),
                             encoding="utf-8")


def pending_status() -> dict:
    """How many labelled feedback examples are waiting for the next round."""
    progress = _load_progress()
    total = 0
    if FEEDBACK_CSV.exists():
        total = sum(1 for _ in open(FEEDBACK_CSV, encoding="utf-8")) - 1
        total = max(total, 0)
    pending = max(total - progress["consumed_feedback_rows"], 0)
    return {"total_feedback": total, "pending_examples": pending,
            "retrain_threshold": RETRAIN_THRESHOLD}


def feedback_examples() -> list[dict]:
    """Turn stored feedback rows into labelled training examples."""
    if not FEEDBACK_CSV.exists():
        return []
    df = pd.read_csv(FEEDBACK_CSV).fillna("")
    rows = []
    for _, r in df.iterrows():
        label = r["correct_category"] if r["label"] == "down" else r["predicted_category"]
        if label in CATEGORIES:
            rows.append({"text": str(r["question"]), "category": label})
    return rows


def retrain_round() -> dict:
    """Train a candidate on old + new training data, score it on the locked
    test set, and replace the current model only if the score is not worse."""
    progress = _load_progress()
    status = pending_status()
    n_new = status["pending_examples"]
    new_rows = feedback_examples()[-n_new:] if n_new else []

    bundle = load_bundle()
    df = load_flashcards()
    train_df, val_df, test_df = stratified_split(df)

    # Guard the locked test set: drop any feedback sentence that duplicates it.
    locked = set(test_df["clean"])
    pool = pd.concat([train_df[["clean", "category"]],
                      val_df[["clean", "category"]]], ignore_index=True)
    added = 0
    for row in new_rows:
        c = clean_text(row["text"])
        if c and c not in locked:
            pool = pd.concat([pool, pd.DataFrame([{"clean": c, "category": row["category"]}])],
                             ignore_index=True)
            added += 1
    pool = pool.drop_duplicates(subset="clean").reset_index(drop=True)

    kind, params = bundle["kind"], bundle["params"]
    vec, clf = _refit(kind, params, pool["clean"], pool["category"])
    metrics = evaluate_on_test(clf, vec, test_df)

    old = bundle["test_metrics"]
    replaced = (metrics["accuracy"], metrics["macro_f1"]) >= (old["accuracy"], old["macro_f1"])

    entry = {
        "round": len(progress["rounds"]) + 1,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "kind": kind,
        "params": params,
        "n_new_examples": added,
        "training_examples": int(len(pool)),
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "previous_test_accuracy": old["accuracy"],
        "replaced": replaced,
        "version": bundle["version"] + 1 if replaced else bundle["version"],
    }
    progress["rounds"].append(entry)
    progress["consumed_feedback_rows"] = status["total_feedback"]

    if replaced:
        new_bundle = dict(bundle)
        new_bundle.update({"version": entry["version"], "vectorizer": vec,
                           "classifier": clf, "test_metrics": metrics,
                           "training_examples": int(len(pool)),
                           "created_at": entry["timestamp"]})
        joblib_dump(new_bundle, MODEL_PATH)
        joblib_dump(new_bundle, VERSIONS_DIR / f"v{entry['version']}.joblib")

    _save_progress(progress)
    return entry


def joblib_dump(obj, path) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def maybe_retrain() -> dict:
    """Start a retraining round if enough new labelled examples are waiting."""
    status = pending_status()
    if status["pending_examples"] >= RETRAIN_THRESHOLD:
        entry = retrain_round()
        return {"retrained": True, "round": entry}
    return {"retrained": False, **status}


def progress_report() -> dict:
    bundle = load_bundle()
    progress = _load_progress()
    status = pending_status()
    return {
        "model_version": bundle["version"],
        "deployed_model": {"kind": bundle["kind"], "params": bundle["params"]},
        "test_accuracy": bundle["test_metrics"]["accuracy"],
        "test_macro_f1": bundle["test_metrics"]["macro_f1"],
        "training_examples": bundle.get("training_examples"),
        "rounds": progress["rounds"],
        "pending_examples": status["pending_examples"],
        "retrain_threshold": RETRAIN_THRESHOLD,
        "total_feedback": status["total_feedback"],
    }
