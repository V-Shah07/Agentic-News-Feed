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

    # Path the Phase 6 promotion step writes the winning model to.
    PROMOTED_MODEL_DIR = "models/finetuned/production"

    def __init__(self, model_name_or_path: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        settings = get_settings()
        # Resolution order: explicit arg > EMBEDDING_MODEL_PATH > promoted model
        # dir (MLflow-promoted) > base model.
        source = (
            model_name_or_path
            or settings.embedding_model_path
            or (self.PROMOTED_MODEL_DIR if os.path.isdir(self.PROMOTED_MODEL_DIR) else "")
            or settings.embedding_model
        )
        self.source = source
        self.is_finetuned = os.path.isdir(str(source))
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
