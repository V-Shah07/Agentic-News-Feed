"""In-memory test doubles so novelty/relevance logic runs without heavy deps."""
from __future__ import annotations

import math


class FakeEmbedder:
    """Deterministic hash-free embedder: maps text to a small vector by keyword.

    Vectors are L2-normalized so dot product == cosine similarity, matching the
    real Embedder's contract.
    """

    source = "fake-embedder"
    is_finetuned = False

    def __init__(self, vocab: dict[str, list[float]]):
        self.vocab = vocab

    def _vec(self, text: str) -> list[float]:
        text_l = text.lower()
        for key, vec in self.vocab.items():
            if key in text_l:
                return _normalize(vec)
        return _normalize([1.0, 0.0, 0.0])

    def encode(self, texts):
        return [self._vec(t) for t in texts]

    def encode_one(self, text):
        return self._vec(text)


class InMemoryVectorStore:
    """Cosine-similarity store backed by a list; mirrors ChromaVectorStore API."""

    def __init__(self):
        self.items: list[tuple[str, list[float], dict, str]] = []

    def add(self, ids, embeddings, metadatas, documents):
        for i, cid in enumerate(ids):
            self.items = [it for it in self.items if it[0] != cid]  # upsert
            self.items.append((cid, embeddings[i], metadatas[i], documents[i]))

    def query(self, embedding, n_results=5, where=None):
        scored = []
        for cid, emb, meta, doc in self.items:
            if where and not _match_where(meta, where):
                continue
            scored.append((cid, _cosine(embedding, emb), meta, doc))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n_results]

    def count(self):
        return len(self.items)


def _normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def _match_where(meta, where):
    for key, cond in where.items():
        val = meta.get(key)
        if isinstance(cond, dict):
            for op, target in cond.items():
                if op == "$gte" and not (val is not None and val >= target):
                    return False
                if op == "$lte" and not (val is not None and val <= target):
                    return False
                if op == "$eq" and val != target:
                    return False
        elif val != cond:
            return False
    return True
