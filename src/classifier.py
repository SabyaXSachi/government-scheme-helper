"""Supervised category classifier: TF-IDF features + KNN / decision tree.

Implements the ML methodology of the project document (section 8):
  - stratified 70/15/15 hold-out split with a locked test set
  - hyperparameter tuning on the validation set only
  - stratified 5-fold cross-validation to check score stability
  - overfitting analysis (train vs validation curves against K and depth)
  - accuracy, per-class precision/recall/F1 and a confusion matrix
"""
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix)

from .config import (CATEGORIES, FLASHCARDS_CSV, MODEL_PATH, REPORTS_DIR,
                     VERSIONS_DIR, RANDOM_STATE)

K_VALUES = [1, 3, 5, 7, 9, 11, 13]
DEPTH_VALUES = [2, 3, 4, 5, 6, 8, 10, None]
MIN_SPLIT_VALUES = [2, 5, 10]


def load_flashcards() -> pd.DataFrame:
    from .text_utils import clean_text
    df = pd.read_csv(FLASHCARDS_CSV)
    df["clean"] = df["text"].map(clean_text)
    return df


def stratified_split(df: pd.DataFrame):
    """Deterministic 70/15/15 stratified split. Same input file + seed
    always reproduces the same test set, which is what keeps it locked."""
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["category"], random_state=RANDOM_STATE)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["category"], random_state=RANDOM_STATE)
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def _fit_features(train_texts, other_texts=None):
    # The vectorizer is fitted on training text only, so validation and test
    # features never leak into the training vocabulary or IDF weights.
    # Stop words are removed (the doc's "optionally remove stop words").
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    X_train = vec.fit_transform(train_texts)
    X_other = vec.transform(other_texts) if other_texts is not None else None
    return vec, X_train, X_other


def _make_classifier(kind: str, params: dict):
    if kind == "knn":
        return KNeighborsClassifier(n_neighbors=params["k"], metric=params["metric"])
    return DecisionTreeClassifier(max_depth=params["max_depth"],
                                  min_samples_split=params["min_samples_split"],
                                  random_state=RANDOM_STATE)


def tune_knn(train_df, val_df) -> tuple[dict, list[dict]]:
    vec, Xtr, Xv = _fit_features(train_df["clean"], val_df["clean"])
    ytr, yv = train_df["category"], val_df["category"]
    results = []
    for k in K_VALUES:
        for metric in ["euclidean", "cosine"]:
            clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
            clf.fit(Xtr, ytr)
            results.append({"k": k, "metric": metric,
                            "val_accuracy": round(accuracy_score(yv, clf.predict(Xv)), 4)})
    best = max(results, key=lambda r: (r["val_accuracy"], -r["k"]))
    params = {"k": best["k"], "metric": best["metric"]}
    return params, results


def tune_dt(train_df, val_df) -> tuple[dict, list[dict]]:
    vec, Xtr, Xv = _fit_features(train_df["clean"], val_df["clean"])
    ytr, yv = train_df["category"], val_df["category"]
    results = []
    for depth in DEPTH_VALUES:
        for min_split in MIN_SPLIT_VALUES:
            clf = DecisionTreeClassifier(max_depth=depth, min_samples_split=min_split,
                                         random_state=RANDOM_STATE)
            clf.fit(Xtr, ytr)
            results.append({"max_depth": depth, "min_samples_split": min_split,
                            "val_accuracy": round(accuracy_score(yv, clf.predict(Xv)), 4)})
    best = max(results, key=lambda r: r["val_accuracy"])
    params = {"max_depth": best["max_depth"], "min_samples_split": best["min_samples_split"]}
    return params, results


def cross_val_check(trainval_df, kind: str, params: dict) -> dict:
    """Stratified 5-fold CV on train+validation data (vectorizer refitted
    inside each fold via a Pipeline, so no fold leaks into another)."""
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")),
        ("clf", _make_classifier(kind, params)),
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, trainval_df["clean"], trainval_df["category"],
                             cv=skf, scoring="accuracy")
    return {"mean": round(float(scores.mean()), 4), "std": round(float(scores.std()), 4)}


def overfitting_curves(train_df, val_df) -> dict:
    """Train vs validation accuracy against K (KNN) and max depth (tree)."""
    vec, Xtr, Xv = _fit_features(train_df["clean"], val_df["clean"])
    ytr, yv = train_df["category"], val_df["category"]

    knn_curve = []
    for k in K_VALUES:
        clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        clf.fit(Xtr, ytr)
        knn_curve.append({"k": k,
                          "train_accuracy": round(accuracy_score(ytr, clf.predict(Xtr)), 4),
                          "val_accuracy": round(accuracy_score(yv, clf.predict(Xv)), 4)})

    dt_curve = []
    for depth in [d for d in DEPTH_VALUES if d is not None] + [12, 16, 20]:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        clf.fit(Xtr, ytr)
        dt_curve.append({"max_depth": depth,
                         "train_accuracy": round(accuracy_score(ytr, clf.predict(Xtr)), 4),
                         "val_accuracy": round(accuracy_score(yv, clf.predict(Xv)), 4)})
    return {"knn_curve": knn_curve, "dt_curve": dt_curve}


def evaluate_on_test(clf, vec, test_df) -> dict:
    X = vec.transform(test_df["clean"])
    y = test_df["category"]
    pred = clf.predict(X)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, labels=CATEGORIES, zero_division=0)
    per_class = {cat: {"precision": round(float(p), 4), "recall": round(float(r), 4),
                       "f1": round(float(f), 4)}
                 for cat, p, r, f in zip(CATEGORIES, precision, recall, f1)}
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y, pred, labels=CATEGORIES).tolist(),
        "labels": CATEGORIES,
    }


def _refit(kind: str, params: dict, train_texts, train_labels):
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    X = vec.fit_transform(train_texts)
    clf = _make_classifier(kind, params)
    clf.fit(X, train_labels)
    return vec, clf


def _save_plots(bundle: dict, curves: dict, out_dir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Overfitting curves: accuracy vs K and vs tree depth.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ks = [p["k"] for p in curves["knn_curve"]]
    axes[0].plot(ks, [p["train_accuracy"] for p in curves["knn_curve"]], "o-", label="train")
    axes[0].plot(ks, [p["val_accuracy"] for p in curves["knn_curve"]], "s-", label="validation")
    axes[0].set_xlabel("K (neighbours)"); axes[0].set_ylabel("accuracy")
    axes[0].set_title("KNN: under/over-fitting"); axes[0].legend(); axes[0].grid(alpha=0.3)

    depths = [p["max_depth"] for p in curves["dt_curve"]]
    axes[1].plot(depths, [p["train_accuracy"] for p in curves["dt_curve"]], "o-", label="train")
    axes[1].plot(depths, [p["val_accuracy"] for p in curves["dt_curve"]], "s-", label="validation")
    axes[1].set_xlabel("max depth"); axes[1].set_ylabel("accuracy")
    axes[1].set_title("Decision tree: under/over-fitting"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "overfitting_curves.png", dpi=150)
    plt.close(fig)

    # Confusion matrix of the deployed model on the test set.
    cm = np.array(bundle["test_metrics"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CATEGORIES)), CATEGORIES, rotation=30, ha="right")
    ax.set_yticks(range(len(CATEGORIES)), CATEGORIES)
    for i in range(len(CATEGORIES)):
        for j in range(len(CATEGORIES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion matrix — test set ({bundle['kind']}, v{bundle['version']})")
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def train_full() -> dict:
    """Run the whole offline pipeline and save the model bundle v1."""
    df = load_flashcards()
    train_df, val_df, test_df = stratified_split(df)
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    print(f"Split: {len(train_df)} train / {len(val_df)} validation / {len(test_df)} test (locked)")

    knn_params, knn_grid = tune_knn(train_df, val_df)
    dt_params, dt_grid = tune_dt(train_df, val_df)
    print(f"Tuned KNN (on validation): {knn_params}")
    print(f"Tuned decision tree (on validation): {dt_params}")

    curves = overfitting_curves(train_df, val_df)

    cv = {"knn": cross_val_check(trainval_df, "knn", knn_params),
          "dt": cross_val_check(trainval_df, "dt", dt_params)}

    # Final test of BOTH models on the locked test set (classifier comparison),
    # then deploy the winner.
    candidates = {}
    for kind, params in [("knn", knn_params), ("dt", dt_params)]:
        vec, clf = _refit(kind, params, trainval_df["clean"], trainval_df["category"])
        candidates[kind] = {"vec": vec, "clf": clf, "params": params,
                            "metrics": evaluate_on_test(clf, vec, test_df)}
        print(f"Test {kind}: accuracy={candidates[kind]['metrics']['accuracy']}, "
              f"macro F1={candidates[kind]['metrics']['macro_f1']}")

    winner = max(candidates, key=lambda k: (candidates[k]["metrics"]["accuracy"],
                                            candidates[k]["metrics"]["macro_f1"]))
    win = candidates[winner]

    bundle = {
        "version": 1,
        "kind": winner,
        "params": win["params"],
        "vectorizer": win["vec"],
        "classifier": win["clf"],
        "test_texts": test_df["clean"].tolist(),
        "test_labels": test_df["category"].tolist(),
        "test_metrics": win["metrics"],
        "tuning": {"knn": {"params": knn_params, "grid": knn_grid},
                   "dt": {"params": dt_params, "grid": dt_grid}},
        "cv": cv,
        "overfitting": curves,
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "training_examples": int(len(trainval_df)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    joblib.dump(bundle, VERSIONS_DIR / "v1.joblib")

    report = {
        "split_sizes": bundle["split_sizes"],
        "tuning": bundle["tuning"],
        "cv": bundle["cv"],
        "test_comparison": {k: {"accuracy": v["metrics"]["accuracy"],
                                "macro_f1": v["metrics"]["macro_f1"],
                                "per_class": v["metrics"]["per_class"]}
                            for k, v in candidates.items()},
        "deployed_model": {"kind": winner, "params": win["params"], "version": 1},
        "test_metrics": win["metrics"],
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "classifier_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _save_plots(bundle, curves, REPORTS_DIR)

    print(f"Deployed model: {winner} {win['params']} (v1) — "
          f"test accuracy {win['metrics']['accuracy']}")
    return bundle


def load_bundle() -> dict:
    return joblib.load(MODEL_PATH)


def predict_category(bundle: dict, question: str) -> tuple[str, float]:
    from .text_utils import clean_text
    x = bundle["vectorizer"].transform([clean_text(question)])
    label = bundle["classifier"].predict(x)[0]
    proba = bundle["classifier"].predict_proba(x)[0]
    confidence = float(np.max(proba)) if proba is not None and len(proba) else 0.0
    return str(label), round(confidence, 4)
