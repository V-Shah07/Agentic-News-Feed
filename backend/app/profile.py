"""Phase 3 — personalized interest profile with an implicit-feedback loop.

The profile is a single vector in embedding space. Implicit feedback nudges it:
'useful' articles pull it toward them, 'skipped' articles push it away. An
article's relevance is its cosine similarity to the (normalized) profile vector.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Article, Feedback, UserProfile

logger = logging.getLogger(__name__)

DEFAULT_PROFILE_ID = "default"
LR_USEFUL = 1.0
LR_SKIPPED = 0.5


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    # inputs are stored already-normalized; renormalize defensively
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class InterestProfile:
    """Mutable interest vector with feedback-driven updates."""

    def __init__(self, vector: list[float] | None = None):
        self.vector: list[float] = list(vector) if vector else []
        self.n_useful = 0
        self.n_skipped = 0

    @property
    def dim(self) -> int:
        return len(self.vector)

    def _ensure_dim(self, d: int) -> None:
        if not self.vector:
            self.vector = [0.0] * d

    def update(self, embedding: list[float], label: str,
               lr_useful: float = LR_USEFUL, lr_skipped: float = LR_SKIPPED) -> None:
        self._ensure_dim(len(embedding))
        if label == "useful":
            self.vector = [v + lr_useful * e for v, e in zip(self.vector, embedding)]
            self.n_useful += 1
        elif label == "skipped":
            self.vector = [v - lr_skipped * e for v, e in zip(self.vector, embedding)]
            self.n_skipped += 1
        else:
            raise ValueError(f"unknown label: {label}")
        # keep on the unit sphere so relevance stays a clean cosine
        self.vector = _normalize(self.vector)

    def score(self, embedding: list[float]) -> float:
        if not self.vector or all(v == 0 for v in self.vector):
            return 0.0
        return _cosine(self.vector, embedding)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def load_profile(session: Session, profile_id: str = DEFAULT_PROFILE_ID) -> InterestProfile:
    row = session.get(UserProfile, profile_id)
    prof = InterestProfile()
    if row and row.vector_json:
        prof.vector = json.loads(row.vector_json)
        prof.n_useful = row.n_useful
        prof.n_skipped = row.n_skipped
    return prof


def save_profile(session: Session, prof: InterestProfile,
                 profile_id: str = DEFAULT_PROFILE_ID) -> None:
    row = session.get(UserProfile, profile_id)
    if row is None:
        row = UserProfile(id=profile_id)
        session.add(row)
    row.vector_json = json.dumps(prof.vector)
    row.dim = prof.dim
    row.n_useful = prof.n_useful
    row.n_skipped = prof.n_skipped
    row.updated_at = dt.datetime.now(dt.timezone.utc)
    session.commit()


def apply_feedback(session: Session, embedder, article: Article, label: str,
                   profile_id: str = DEFAULT_PROFILE_ID) -> InterestProfile:
    """Record feedback and nudge the persisted profile in one step."""
    from .embeddings import embedding_text

    prof = load_profile(session, profile_id)
    emb = embedder.encode_one(embedding_text(article.title, article.content))
    prof.update(emb, label)
    session.add(Feedback(article_id=article.id, label=label))
    save_profile(session, prof, profile_id)
    return prof


def score_all_articles(session: Session, embedder,
                       profile_id: str = DEFAULT_PROFILE_ID) -> int:
    """Recompute relevance_score for every article against the current profile."""
    prof = load_profile(session, profile_id)
    if prof.dim == 0:
        return 0
    from .embeddings import embedding_text

    articles = list(session.scalars(select(Article)))
    for art in articles:
        emb = embedder.encode_one(embedding_text(art.title, art.content))
        art.relevance_score = prof.score(emb)
    session.commit()
    return len(articles)
