"""FastAPI application entrypoint."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import get_settings
from .db import get_session, init_db
from .ingest import run_ingest
from .ingest.sources import source_count
from .models import Article, Feedback
from .schemas import (
    ArticleOut,
    FeedbackIn,
    HealthResponse,
    IngestResponse,
)
from .scheduler import shutdown_scheduler, start_scheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    settings = get_settings()
    if settings.ingest_interval_seconds > 0:
        start_scheduler()
    yield
    shutdown_scheduler()


app = FastAPI(title="Agentic Information Diet Manager", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health(session: Session = Depends(get_session)) -> HealthResponse:
    count = session.scalar(select(func.count(Article.id))) or 0
    return HealthResponse(status="ok", articles=count, sources_configured=source_count())


@app.post("/ingest", response_model=IngestResponse)
def ingest(session: Session = Depends(get_session)) -> IngestResponse:
    result = run_ingest(session)
    return IngestResponse(
        sources_hit=result.sources_hit,
        fetched=result.fetched,
        inserted=result.inserted,
        duplicates=result.duplicates,
        per_source=result.per_source,
    )


@app.get("/articles", response_model=list[ArticleOut])
def list_articles(
    limit: int = 50,
    offset: int = 0,
    source: str | None = None,
    session: Session = Depends(get_session),
) -> list[Article]:
    stmt = select(Article).order_by(Article.fetched_at.desc())
    if source:
        stmt = stmt.where(Article.source == source)
    stmt = stmt.limit(min(limit, 200)).offset(offset)
    return list(session.scalars(stmt))


@app.post("/feedback")
def submit_feedback(payload: FeedbackIn, session: Session = Depends(get_session)) -> dict:
    if payload.label not in ("useful", "skipped"):
        raise HTTPException(status_code=400, detail="label must be 'useful' or 'skipped'")
    article = session.get(Article, payload.article_id)
    if not article:
        raise HTTPException(status_code=404, detail="article not found")
    session.add(Feedback(article_id=payload.article_id, label=payload.label))
    session.commit()
    return {"status": "ok"}
