"""Vector-store abstraction with a ChromaDB backend.

The novelty filter, interest profile, and RAG retrieval all talk to this small
interface, so the logic is unit-testable with an in-memory fake while the real
pipeline uses Chroma (persistent locally, HTTP against the `chromadb` service in
Docker Compose).
"""
from __future__ import annotations

import logging
from typing import Protocol

from .config import get_settings

logger = logging.getLogger(__name__)

ARTICLES_COLLECTION = "articles"
PROFILE_COLLECTION = "interest_profile"


class VectorStore(Protocol):
    def add(self, ids: list[str], embeddings: list[list[float]], metadatas: list[dict],
            documents: list[str]) -> None: ...

    def query(self, embedding: list[float], n_results: int = 5,
              where: dict | None = None) -> list[tuple[str, float, dict, str]]:
        """Return [(id, cosine_similarity, metadata, document), ...] sorted desc."""
        ...

    def count(self) -> int: ...


class ChromaVectorStore:
    """Cosine-space Chroma collection wrapper."""

    def __init__(self, collection_name: str = ARTICLES_COLLECTION, client=None):
        import chromadb

        settings = get_settings()
        if client is not None:
            self.client = client
        elif settings.chroma_host and settings.chroma_host not in ("", "local"):
            try:
                self.client = chromadb.HttpClient(
                    host=settings.chroma_host, port=settings.chroma_port
                )
                self.client.heartbeat()
            except Exception as exc:  # fall back to local persistence
                logger.warning("Chroma HTTP unavailable (%s); using PersistentClient", exc)
                self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        else:
            self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids, embeddings, metadatas, documents) -> None:
        self.collection.upsert(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )

    def query(self, embedding, n_results=5, where=None):
        n = min(n_results, max(self.collection.count(), 1))
        res = self.collection.query(
            query_embeddings=[embedding],
            n_results=n,
            where=where,
            include=["distances", "metadatas", "documents"],
        )
        out: list[tuple[str, float, dict, str]] = []
        if not res["ids"] or not res["ids"][0]:
            return out
        for i, cid in enumerate(res["ids"][0]):
            dist = res["distances"][0][i]
            sim = 1.0 - float(dist)  # cosine distance -> similarity
            meta = (res["metadatas"][0][i] or {}) if res.get("metadatas") else {}
            doc = res["documents"][0][i] if res.get("documents") else ""
            out.append((cid, sim, meta, doc))
        return out

    def count(self) -> int:
        return self.collection.count()


def get_articles_store() -> ChromaVectorStore:
    return ChromaVectorStore(ARTICLES_COLLECTION)
