"""Phase 2 — semantic novelty / deduplication filter.

For each embedded article we query the vector store for its nearest neighbour
within a rolling window. If cosine similarity >= threshold the article is flagged
redundant. Every article is still stored (RAG in Phase 5 needs them all); the
flag drives the dedup metric.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .embeddings import embedding_text
from .models import Article

logger = logging.getLogger(__name__)


@dataclass
class NoveltyResult:
    processed: int = 0
    redundant: int = 0
    novel: int = 0

    @property
    def reduction_pct(self) -> float:
        return (self.redundant / self.processed * 100.0) if self.processed else 0.0

    def summary(self) -> str:
        return (
            f"ingested={self.processed} filtered_redundant={self.redundant} "
            f"novel={self.novel} reduction={self.reduction_pct:.1f}%"
        )


def _ts_epoch(article: Article) -> int:
    when = article.published_at or article.fetched_at or dt.datetime.now(dt.timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    return int(when.timestamp())


def filter_batch(
    session: Session,
    embedder,
    store,
    threshold: float | None = None,
    window_days: int | None = None,
    only_unembedded: bool = True,
) -> NoveltyResult:
    """Embed pending articles, flag redundant ones, persist embeddings + flags."""
    settings = get_settings()
    threshold = settings.novelty_threshold if threshold is None else threshold
    window_days = settings.novelty_window_days if window_days is None else window_days

    stmt = select(Article).order_by(Article.published_at.asc().nullslast(), Article.id.asc())
    if only_unembedded:
        stmt = stmt.where(Article.embedded.is_(False))
    articles = list(session.scalars(stmt))

    result = NoveltyResult()
    now = dt.datetime.now(dt.timezone.utc)
    window_start = int((now - dt.timedelta(days=window_days)).timestamp())

    for art in articles:
        text = embedding_text(art.title, art.content)
        emb = embedder.encode_one(text)

        neighbours = store.query(
            emb, n_results=1, where={"ts": {"$gte": window_start}}
        )
        max_sim = neighbours[0][1] if neighbours else 0.0
        is_redundant = bool(neighbours) and max_sim >= threshold

        store.add(
            ids=[str(art.id)],
            embeddings=[emb],
            metadatas=[
                {
                    "article_id": art.id,
                    "source": art.source,
                    "source_type": art.source_type,
                    "ts": _ts_epoch(art),
                    "is_redundant": is_redundant,
                }
            ],
            documents=[text],
        )

        art.embedded = True
        art.max_similarity = float(max_sim)
        art.is_redundant = is_redundant

        result.processed += 1
        if is_redundant:
            result.redundant += 1
        else:
            result.novel += 1

    session.commit()
    logger.info("Novelty filter: %s", result.summary())
    return result
