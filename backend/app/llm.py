"""Pluggable LLM backend for cluster synthesis (Phase 4) and RAG answers (Phase 5).

Selection order (best available wins):
  1. OpenAI  — used when OPENAI_API_KEY is set AND a live health check succeeds.
  2. Local transformer — a CPU abstractive summarizer (distilbart-cnn), so
     "synthesis via an LLM" is genuinely generative without any external key.
  3. Extractive — deterministic lead/keyword extraction; always available.

The rest of the codebase talks only to `get_llm()`, so the production backend can
change (e.g. once OpenAI quota is available) with no other edits.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

from .config import get_settings

logger = logging.getLogger(__name__)

_SUMMARY_MODEL = "sshleifer/distilbart-cnn-12-6"


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


class ExtractiveBackend:
    """Always-available fallback: lead-sentence + title synthesis."""

    name = "extractive"
    is_generative = False

    def summarize_cluster(self, titles: list[str], texts: list[str], max_words: int = 90) -> str:
        sentences: list[str] = []
        for t in texts:
            sentences.extend(_split_sentences(t)[:1])
        lead = " ".join(sentences[:3]) if sentences else ""
        headline = titles[0] if titles else "Cluster"
        body = lead or "; ".join(titles[:3])
        words = body.split()
        if len(words) > max_words:
            body = " ".join(words[:max_words]) + "…"
        return f"{headline} — {body}"

    def answer(self, question: str, contexts: list[str], max_words: int = 130) -> str:
        sentences: list[str] = []
        for c in contexts:
            sentences.extend(_split_sentences(c)[:2])
        body = " ".join(sentences[:5])
        words = body.split()
        if len(words) > max_words:
            body = " ".join(words[:max_words]) + "…"
        return body or "No relevant content retrieved."


class LocalTransformerBackend:
    """CPU abstractive summarizer via HuggingFace transformers."""

    name = f"local:{_SUMMARY_MODEL}"
    is_generative = True

    def __init__(self):
        from transformers import pipeline  # lazy heavy import

        logger.info("Loading local summarization model: %s", _SUMMARY_MODEL)
        self._pipe = pipeline("summarization", model=_SUMMARY_MODEL, device=-1)

    def _summarize(self, text: str, max_len: int, min_len: int) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""
        # distilbart handles ~1024 tokens; truncate defensively by characters.
        text = text[:3000]
        out = self._pipe(text, max_length=max_len, min_length=min_len, truncation=True)
        return out[0]["summary_text"].strip()

    def summarize_cluster(self, titles: list[str], texts: list[str], max_words: int = 90) -> str:
        joined = " ".join(f"{t}. {x}" for t, x in zip(titles, texts))
        return self._summarize(joined, max_len=110, min_len=30)

    def answer(self, question: str, contexts: list[str], max_words: int = 130) -> str:
        joined = f"{question} " + " ".join(contexts)
        return self._summarize(joined, max_len=160, min_len=45)


class OpenAIBackend:
    """OpenAI chat-completions backend (preferred when quota is available)."""

    is_generative = True

    def __init__(self):
        from openai import OpenAI

        settings = get_settings()
        self.model = settings.openai_model
        self.name = f"openai:{self.model}"
        self._client = OpenAI(api_key=settings.openai_api_key)

    def _chat(self, system: str, user: str, max_tokens: int) -> str:
        r = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return r.choices[0].message.content.strip()

    def summarize_cluster(self, titles: list[str], texts: list[str], max_words: int = 90) -> str:
        joined = "\n".join(f"- {t}: {x[:400]}" for t, x in zip(titles, texts))
        return self._chat(
            "You are a concise news editor. Synthesize the related stories into one "
            "briefing of 2-3 sentences. No preamble.",
            f"Related stories:\n{joined}",
            max_tokens=180,
        )

    def answer(self, question: str, contexts: list[str], max_words: int = 130) -> str:
        ctx = "\n\n".join(f"[{i+1}] {c[:600]}" for i, c in enumerate(contexts))
        return self._chat(
            "Answer the question using only the provided articles. Cite sources as [n]. "
            "Be specific and concise.",
            f"Question: {question}\n\nArticles:\n{ctx}",
            max_tokens=320,
        )


def _openai_healthy() -> bool:
    settings = get_settings()
    if not settings.has_openai:
        return False
    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return True
    except Exception as exc:
        logger.warning("OpenAI unavailable (%s); using local/extractive backend", type(exc).__name__)
        return False


@lru_cache
def get_llm(prefer_local: bool = False):
    """Return the best available LLM backend."""
    if not prefer_local and _openai_healthy():
        logger.info("LLM backend: OpenAI")
        return OpenAIBackend()
    try:
        backend = LocalTransformerBackend()
        logger.info("LLM backend: %s", backend.name)
        return backend
    except Exception as exc:
        logger.warning("Local transformer unavailable (%s); using extractive backend", exc)
        return ExtractiveBackend()
