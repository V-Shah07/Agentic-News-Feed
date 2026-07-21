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
from .models import Article
from .schemas import (
    ArticleOut,
    FeedbackIn,
    HealthResponse,
    IngestResponse,
    QueryIn,
    QueryResponse,
    SourceOut,
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
    rank_by: str = "recent",  # recent | relevance
    include_redundant: bool = True,
    session: Session = Depends(get_session),
) -> list[Article]:
    stmt = select(Article)
    if source:
        stmt = stmt.where(Article.source == source)
    if not include_redundant:
        stmt = stmt.where(Article.is_redundant.is_(False))
    if rank_by == "relevance":
        stmt = stmt.order_by(Article.relevance_score.desc().nullslast())
    else:
        stmt = stmt.order_by(Article.fetched_at.desc())
    stmt = stmt.limit(min(limit, 200)).offset(offset)
    return list(session.scalars(stmt))


@app.post("/feedback")
def submit_feedback(payload: FeedbackIn, session: Session = Depends(get_session)) -> dict:
    """Record implicit feedback and nudge the interest profile (Phase 3 loop)."""
    if payload.label not in ("useful", "skipped"):
        raise HTTPException(status_code=400, detail="label must be 'useful' or 'skipped'")
    article = session.get(Article, payload.article_id)
    if not article:
        raise HTTPException(status_code=404, detail="article not found")

    from .embeddings import get_embedder
    from .profile import apply_feedback

    prof = apply_feedback(session, get_embedder(), article, payload.label)
    return {"status": "ok", "profile_useful": prof.n_useful, "profile_skipped": prof.n_skipped}


@app.get("/profile")
def get_profile(session: Session = Depends(get_session)) -> dict:
    from .profile import load_profile

    prof = load_profile(session)
    return {"dim": prof.dim, "n_useful": prof.n_useful, "n_skipped": prof.n_skipped}


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryIn, session: Session = Depends(get_session)) -> QueryResponse:
    """RAG: natural-language question -> semantic retrieval -> cited LLM answer."""
    from .embeddings import get_embedder
    from .llm import get_llm
    from .rag import answer_query
    from .vectorstore import get_articles_store

    result = answer_query(
        session, get_embedder(), get_articles_store(), get_llm(),
        payload.query, k=payload.k, window_days=payload.window_days,
    )
    return QueryResponse(
        query=result.query,
        answer=result.answer,
        sources=[SourceOut(**s.__dict__) for s in result.sources],
    )
