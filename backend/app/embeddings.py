"""Sentence-transformer embedding backend.

Loads a base `all-MiniLM-L6-v2` model by default. When Phase 6 promotes a
fine-tuned model, EMBEDDING_MODEL_PATH points the production pipeline at it —
no other code changes.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from .config import get_settings

logger = logging.getLogger(__name__)


def embedding_text(title: str, content: str = "", max_chars: int = 800) -> str:
    """Compose the text that represents an article for embedding."""
    title = (title or "").strip()
    content = (content or "").strip()
    text = f"{title}. {content}" if content else title
    return text[:max_chars]


class Embedder:
    """Thin wrapper around a SentenceTransformer with cosine-normalized output."""

    def __init__(self, model_name_or_path: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        settings = get_settings()
        source = model_name_or_path or settings.embedding_model_path or settings.embedding_model
        self.source = source
        self.is_finetuned = bool(
            settings.embedding_model_path and source == settings.embedding_model_path
        ) or bool(model_name_or_path and os.path.exists(str(model_name_or_path)))
        logger.info("Loading embedding model: %s", source)
        self.model = SentenceTransformer(source)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,  # cosine similarity == dot product
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()
