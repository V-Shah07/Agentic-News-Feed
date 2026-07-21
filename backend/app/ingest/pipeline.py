"""Ingestion pipeline: fetch every source, dedupe by URL, persist to Postgres."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import Article
from . import adapters
from .sources import ALL_RSS_LIKE

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    fetched: int = 0
    inserted: int = 0
    duplicates: int = 0
    per_source: dict[str, int] = field(default_factory=dict)
    sources_hit: int = 0

    def summary(self) -> str:
        return (
            f"sources_hit={self.sources_hit} fetched={self.fetched} "
            f"inserted={self.inserted} duplicates={self.duplicates}"
        )


def _persist(session: Session, records: list[dict], result: IngestResult) -> None:
    for rec in records:
        url = rec.get("url")
        if not url:
            continue
        result.fetched += 1
        exists = session.scalar(select(Article.id).where(Article.url == url))
        if exists:
            result.duplicates += 1
            continue
        session.add(Article(**rec))
        try:
            session.flush()  # surface unique violations within the batch
            result.inserted += 1
            result.per_source[rec["source"]] = result.per_source.get(rec["source"], 0) + 1
        except Exception:
            session.rollback()
            result.duplicates += 1


def run_ingest(session: Session, limit: int | None = None) -> IngestResult:
    """Fetch from all configured sources and persist new articles."""
    settings = get_settings()
    per_source_limit = limit or settings.ingest_per_source_limit
    result = IngestResult()

    # HackerNews
    hn = adapters.fetch_hackernews(limit=per_source_limit)
    if hn:
        result.sources_hit += 1
    _persist(session, hn, result)
    session.commit()

    # RSS + Reddit RSS
    for src in ALL_RSS_LIKE:
        records = adapters.fetch_rss(src, limit=per_source_limit)
        if records:
            result.sources_hit += 1
        _persist(session, records, result)
        session.commit()

    logger.info("Ingest complete: %s", result.summary())
    return result
