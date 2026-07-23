"""Phase 5 — RAG over the reading history.

A natural-language question is embedded, the nearest article chunks are retrieved
from ChromaDB, and the LLM synthesizes a cited answer grounded in them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from .embeddings import embedding_text
from .models import Article

logger = logging.getLogger(__name__)


@dataclass
class RetrievedArticle:
    article_id: int
    similarity: float
    title: str
    source: str
    url: str
    snippet: str


@dataclass
class RagAnswer:
    query: str
    answer: str
    sources: list[RetrievedArticle] = field(default_factory=list)


def retrieve(session: Session, embedder, store, query: str, k: int = 6,
             window_days: int | None = None) -> list[RetrievedArticle]:
    q_emb = embedder.encode_one(query)
    where = None
    if window_days:
        import datetime as dt

        cutoff = int((dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=window_days)).timestamp())
        where = {"ts": {"$gte": cutoff}}

    hits = store.query(q_emb, n_results=k, where=where)
    out: list[RetrievedArticle] = []
    for cid, sim, meta, doc in hits:
        aid = int(meta.get("article_id", cid)) if meta else int(cid)
        art = session.get(Article, aid)
        if art is None:
            continue
        out.append(
            RetrievedArticle(
                article_id=aid,
                similarity=round(float(sim), 4),
                title=art.title,
                source=art.source,
                url=art.url,
                snippet=(doc or embedding_text(art.title, art.content))[:280],
            )
        )
    return out


def answer_query(session: Session, embedder, store, llm, query: str,
                 k: int = 6, window_days: int | None = None) -> RagAnswer:
    sources = retrieve(session, embedder, store, query, k=k, window_days=window_days)
    if not sources:
        return RagAnswer(query=query, answer="No relevant articles were found for this query.")
    contexts = [f"{s.title}. {s.snippet}" for s in sources]
    answer = llm.answer(query, contexts)
    logger.info("RAG answered %r with %d sources", query, len(sources))
    return RagAnswer(query=query, answer=answer, sources=sources)
