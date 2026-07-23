"""Pydantic response/request schemas."""
from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict


class ArticleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source: str
    source_type: str
    title: str
    url: str
    author: str | None = None
    published_at: dt.datetime | None = None
    fetched_at: dt.datetime | None = None
    is_redundant: bool = False
    relevance_score: float | None = None


class IngestResponse(BaseModel):
    sources_hit: int
    fetched: int
    inserted: int
    duplicates: int
    per_source: dict[str, int]


class FeedbackIn(BaseModel):
    article_id: int
    label: str  # useful | skipped


class HealthResponse(BaseModel):
    status: str
    articles: int
    sources_configured: int


class QueryIn(BaseModel):
    query: str
    k: int = 6
    window_days: int | None = None


class SourceOut(BaseModel):
    article_id: int
    similarity: float
    title: str
    source: str
    url: str
    snippet: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceOut]
