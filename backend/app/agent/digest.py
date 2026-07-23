"""LangChain digest agent.

The agent decides *dynamically* which tools to invoke based on article volume:
  - low volume  -> skip clustering, summarize the batch directly.
  - high volume -> cluster_tool first, then summarize_tool once per cluster.
Every tool invocation is recorded in a trace so the "agent chooses tools"
behaviour is auditable (this backs the agentic resume bullet).

When a working OpenAI key is present, an LLM-driven tool-calling agent can drive
the same tools; absent that, a volume policy selects them. Either way the tools
are real LangChain StructuredTools invoked via `.invoke(...)`.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..embeddings import embedding_text, get_embedder
from ..llm import get_llm
from ..models import Article
from .tools import build_cluster_tool, build_summarize_tool, choose_k

logger = logging.getLogger(__name__)

# Below this many articles, clustering adds no value — summarize directly.
CLUSTER_VOLUME_THRESHOLD = 6


@dataclass
class ToolCall:
    tool: str
    args_summary: str
    output_summary: str


@dataclass
class ClusterBriefing:
    cluster_id: int
    size: int
    article_ids: list[int]
    top_titles: list[str]
    briefing: str


@dataclass
class Digest:
    generated_at: str
    n_articles: int
    strategy: str          # "cluster+summarize" | "summarize-only"
    n_clusters: int
    briefings: list[ClusterBriefing] = field(default_factory=list)
    trace: list[ToolCall] = field(default_factory=list)


class DigestAgent:
    def __init__(self, llm=None, embedder=None):
        self.llm = llm or get_llm()
        self.embedder = embedder or get_embedder()
        self.cluster_tool = build_cluster_tool()
        self.summarize_tool = build_summarize_tool(self.llm)

    def _load_articles(self, session: Session, since_hours: int, exclude_redundant: bool,
                       limit: int) -> list[Article]:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=since_hours)
        stmt = select(Article)
        if exclude_redundant:
            stmt = stmt.where(Article.is_redundant.is_(False))
        stmt = stmt.order_by(Article.relevance_score.desc().nullslast(),
                             Article.fetched_at.desc()).limit(limit)
        return list(session.scalars(stmt))

    def run(self, session: Session, since_hours: int = 48, exclude_redundant: bool = True,
            limit: int = 120) -> Digest:
        articles = self._load_articles(session, since_hours, exclude_redundant, limit)
        n = len(articles)
        digest = Digest(
            generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            n_articles=n, strategy="", n_clusters=0,
        )
        if n == 0:
            digest.strategy = "empty"
            return digest

        texts = [embedding_text(a.title, a.content) for a in articles]
        embeddings = self.embedder.encode(texts)

        # --- agentic decision: volume drives tool selection ---
        if n >= CLUSTER_VOLUME_THRESHOLD:
            digest.strategy = "cluster+summarize"
            k = choose_k(n)
            logger.info("Agent decision: %d articles >= %d -> cluster_tool (k=%d) then summarize",
                        n, CLUSTER_VOLUME_THRESHOLD, k)
            labels = self.cluster_tool.invoke({"embeddings": embeddings, "k": k})
            digest.trace.append(ToolCall(
                "cluster_tool", f"{n} embeddings, k={k}",
                f"{len(set(labels))} clusters",
            ))
        else:
            digest.strategy = "summarize-only"
            logger.info("Agent decision: %d articles < %d -> summarize directly (no clustering)",
                        n, CLUSTER_VOLUME_THRESHOLD)
            labels = [0] * n

        # group
        groups: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels):
            groups.setdefault(lab, []).append(idx)
        digest.n_clusters = len(groups)

        # --- summarize each group with the summarize_tool ---
        for cid, idxs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            titles = [articles[i].title for i in idxs]
            bodies = [articles[i].content or articles[i].title for i in idxs]
            briefing = self.summarize_tool.invoke({"titles": titles, "texts": bodies})
            digest.trace.append(ToolCall(
                "summarize_tool", f"cluster {cid}: {len(idxs)} articles",
                briefing[:80] + ("…" if len(briefing) > 80 else ""),
            ))
            digest.briefings.append(ClusterBriefing(
                cluster_id=cid, size=len(idxs),
                article_ids=[articles[i].id for i in idxs],
                top_titles=titles[:4], briefing=briefing,
            ))

        logger.info("Digest: %s, %d clusters, %d tool calls",
                    digest.strategy, digest.n_clusters, len(digest.trace))
        return digest
