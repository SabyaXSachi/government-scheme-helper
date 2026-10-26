"""Retrieval experiment (evaluation plan, item 3).

Baseline RAG searching ALL schemes vs RAG with the category filter, measured
as the share of evaluation questions where the correct scheme appears in the
top 3 results. The classifier itself is also scored on the same questions.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import CATEGORIES, EVALUATION_CSV, REPORTS_DIR, TOP_K
from .classifier import predict_category
from .rag import load_runtime
from .text_utils import clean_text


def run_retrieval_experiment() -> dict:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rt = load_runtime()
    bundle, store, embedder = rt["bundle"], rt["store"], rt["embedder"]

    df = pd_read_eval()
    baseline_hits, filtered_hits, cls_correct = 0, 0, 0
    per_category = {c: {"n": 0, "baseline_top3": 0, "filtered_top3": 0} for c in CATEGORIES}

    for _, row in df.iterrows():
        expected = row["expected_scheme"]
        cat = row["expected_category"]
        qvec = embedder.encode([clean_text(row["question"])])[0]

        predicted_cat, _conf = predict_category(bundle, row["question"])
        base = store.search(qvec, k=TOP_K, category=None)
        filt = store.search(qvec, k=TOP_K, category=predicted_cat)

        base_ok = any(h["metadata"]["scheme_name"] == expected for h in base)
        filt_ok = any(h["metadata"]["scheme_name"] == expected for h in filt)
        baseline_hits += base_ok
        filtered_hits += filt_ok
        cls_correct += predicted_cat == cat
        per_category[cat]["n"] += 1
        per_category[cat]["baseline_top3"] += base_ok
        per_category[cat]["filtered_top3"] += filt_ok

    n = len(df)
    report = {
        "n_questions": n,
        "top_k": TOP_K,
        "classifier_accuracy_on_eval": round(cls_correct / n, 4),
        "baseline_top3_hit_rate": round(baseline_hits / n, 4),
        "filtered_top3_hit_rate": round(filtered_hits / n, 4),
        "improvement": round(filtered_hits / n - baseline_hits / n, 4),
        "per_category": {
            c: {"n": v["n"],
                "baseline_top3": round(v["baseline_top3"] / v["n"], 4) if v["n"] else None,
                "filtered_top3": round(v["filtered_top3"] / v["n"], 4) if v["n"] else None}
            for c, v in per_category.items()},
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "retrieval_experiment.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _plot(report)
    print(f"Baseline RAG top-{TOP_K} hit rate:     {report['baseline_top3_hit_rate']}")
    print(f"Category-filtered top-{TOP_K} hit rate: {report['filtered_top3_hit_rate']}")
    print(f"Improvement: {report['improvement']:+}")
    print(f"Classifier accuracy on evaluation set: {report['classifier_accuracy_on_eval']}")
    return report


def pd_read_eval():
    import pandas as pd
    return pd.read_csv(EVALUATION_CSV)


def _plot(report: dict) -> None:
    cats = list(report["per_category"].keys())
    base = [report["per_category"][c]["baseline_top3"] or 0 for c in cats]
    filt = [report["per_category"][c]["filtered_top3"] or 0 for c in cats]

    x = range(len(cats) + 1)
    base_all = base + [report["baseline_top3_hit_rate"]]
    filt_all = filt + [report["filtered_top3_hit_rate"]]
    labels = list(cats) + ["Overall"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.38
    ax.bar([i - w / 2 for i in x], base_all, w, label="baseline (search all schemes)")
    ax.bar([i + w / 2 for i in x], filt_all, w, label="category-filtered RAG")
    ax.set_xticks(list(x), labels, rotation=20, ha="right")
    ax.set_ylabel(f"top-{report['top_k']} hit rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval experiment: effect of the category model on retrieval")
    ax.legend()
    for i, (b, f) in enumerate(zip(base_all, filt_all)):
        ax.text(i - w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, f + 0.02, f"{f:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "retrieval_experiment.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_retrieval_experiment()
