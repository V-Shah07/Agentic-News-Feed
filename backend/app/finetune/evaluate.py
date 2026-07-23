"""Held-out relevance metric for base-vs-fine-tuned comparison.

Mirrors the production relevance path (Phase 3): build an interest profile as the
mean of TRAIN 'useful' embeddings, then score TEST articles by cosine similarity
to it. Report the ranking AUC (P[useful ranked above skipped]) and the mean
relevance separation.
"""
from __future__ import annotations

import numpy as np

from .dataset import LabeledItem


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def evaluate_model(model, train: list[LabeledItem], test: list[LabeledItem]) -> dict:
    train_useful = [i.text for i in train if i.label == "useful"]
    if not train_useful:
        raise ValueError("no useful training items to build the profile")

    prof = _normalize(np.asarray(
        model.encode(train_useful, normalize_embeddings=True)
    ).mean(axis=0))

    te_useful = [i.text for i in test if i.label == "useful"]
    te_skip = [i.text for i in test if i.label == "skipped"]
    su = np.asarray(model.encode(te_useful, normalize_embeddings=True)) @ prof
    ss = np.asarray(model.encode(te_skip, normalize_embeddings=True)) @ prof

    # ranking AUC over all (useful, skipped) pairs
    pairs = correct = 0
    for u in su:
        for s in ss:
            pairs += 1
            correct += 1.0 if u > s else (0.5 if u == s else 0.0)
    auc = correct / pairs if pairs else 0.0

    return {
        "auc": round(float(auc), 4),
        "mean_relevance_useful": round(float(su.mean()), 4),
        "mean_relevance_skipped": round(float(ss.mean()), 4),
        "separation": round(float(su.mean() - ss.mean()), 4),
        "n_test_useful": len(te_useful),
        "n_test_skipped": len(te_skip),
    }
