"""Custom LangChain tools for the digest agent: cluster_tool + summarize_tool."""
from __future__ import annotations

import logging

import numpy as np
from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)


def choose_k(n: int) -> int:
    """Pick a k-means cluster count from the number of articles."""
    if n < 4:
        return 1
    return max(2, min(12, round(n / 6)))


def cluster_embeddings(embeddings: list[list[float]], k: int | None = None) -> list[int]:
    """k-means over article embeddings; returns a cluster label per article."""
    from sklearn.cluster import KMeans

    n = len(embeddings)
    if n == 0:
        return []
    k = k or choose_k(n)
    k = min(k, n)
    if k <= 1:
        return [0] * n
    X = np.asarray(embeddings, dtype=float)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return labels.tolist()


def build_cluster_tool() -> StructuredTool:
    def _run(embeddings: list[list[float]], k: int | None = None) -> list[int]:
        """Group related articles by k-means over their embeddings."""
        return cluster_embeddings(embeddings, k)

    return StructuredTool.from_function(
        func=_run,
        name="cluster_tool",
        description=(
            "Group a set of article embeddings into clusters of related stories "
            "using k-means. Use when there are enough articles that grouping adds "
            "value. Returns a cluster label for each article."
        ),
    )


def build_summarize_tool(llm) -> StructuredTool:
    def _run(titles: list[str], texts: list[str]) -> str:
        """Synthesize a set of related articles into one short briefing."""
        return llm.summarize_cluster(titles, texts)

    return StructuredTool.from_function(
        func=_run,
        name="summarize_tool",
        description=(
            "Synthesize one or more related articles into a single concise briefing "
            "via the language model. Use once per cluster (or once over all articles "
            "when volume is low)."
        ),
    )
