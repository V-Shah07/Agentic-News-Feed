"""Build fine-tuning data from implicit feedback labels.

'useful'/'skipped' labels become contrastive pairs: same-class pairs are
positives (target similarity 1.0), cross-class pairs are negatives (0.0). This
teaches the embedding space to pull the user's interests together and push
uninteresting content away — i.e. align embeddings to reading history.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..embeddings import embedding_text
from ..feedback_sim import label_title
from ..models import Article

SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "articles_snapshot.json")


@dataclass
class LabeledItem:
    article_id: int
    text: str
    label: str  # useful | skipped


def load_labeled(session: Session) -> list[LabeledItem]:
    items: list[LabeledItem] = []
    for art in session.scalars(select(Article)):
        lb = label_title(art.title)
        if lb is None:
            continue
        items.append(LabeledItem(art.id, embedding_text(art.title, art.content), lb))
    return items


def load_labeled_snapshot(path: str = SNAPSHOT_PATH) -> list[LabeledItem]:
    """Load labeled items from the committed corpus snapshot (no DB required).

    Keeps Phase 6 reproducible and independent of a live Postgres, which matters
    for long training runs and for anyone re-running the fine-tune from a clone.
    """
    rows = json.load(open(os.path.abspath(path)))
    items: list[LabeledItem] = []
    for r in rows:
        lb = label_title(r["title"])
        if lb is None:
            continue
        items.append(LabeledItem(r["id"], embedding_text(r["title"], r.get("content", "")), lb))
    return items


def split(items: list[LabeledItem], test_ratio: float = 0.3, seed: int = 13):
    rng = random.Random(seed)
    useful = [i for i in items if i.label == "useful"]
    skipped = [i for i in items if i.label == "skipped"]
    rng.shuffle(useful)
    rng.shuffle(skipped)

    def cut(xs):
        n_test = max(1, int(len(xs) * test_ratio))
        return xs[n_test:], xs[:n_test]

    tr_u, te_u = cut(useful)
    tr_s, te_s = cut(skipped)
    return (tr_u + tr_s), (te_u + te_s)


def build_pairs(train: list[LabeledItem], max_pairs: int = 1600, seed: int = 13):
    """Return (sentence1, sentence2, label) triples for ContrastiveLoss."""
    rng = random.Random(seed)
    useful = [i.text for i in train if i.label == "useful"]
    skipped = [i.text for i in train if i.label == "skipped"]

    pos: list[tuple[str, str, float]] = []
    neg: list[tuple[str, str, float]] = []

    # cross-class negatives (useful vs skipped)
    for u in useful:
        for s in skipped:
            neg.append((u, s, 0.0))
    rng.shuffle(neg)

    # same-class positives
    def sample_pairs(pool, n):
        out = []
        if len(pool) < 2:
            return out
        for _ in range(n):
            a, b = rng.sample(pool, 2)
            out.append((a, b, 1.0))
        return out

    n_neg = min(len(neg), max_pairs // 2)
    neg = neg[:n_neg]
    pos = sample_pairs(useful, n_neg // 2 + 1) + sample_pairs(skipped, n_neg // 2 + 1)
    rng.shuffle(pos)
    pos = pos[:n_neg]

    pairs = pos + neg
    rng.shuffle(pairs)
    s1 = [p[0] for p in pairs]
    s2 = [p[1] for p in pairs]
    labels = [p[2] for p in pairs]
    return {"sentence1": s1, "sentence2": s2, "label": labels}
