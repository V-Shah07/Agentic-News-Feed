"""Seeded implicit-feedback labels.

Per the spec's de-risk plan, feedback may be simulated when real reading history
is thin. We model a user who is into AI / ML / systems / security and skips
deals, celebrity, sports and lifestyle filler. The labels are keyword-derived
from titles, so they are a legitimate (if noisy) training signal — and we are
honest that they are seeded.
"""
from __future__ import annotations

USEFUL_KEYWORDS = [
    "ai", "artificial intelligence", "machine learning", " ml ", "llm", "gpt",
    "openai", "anthropic", "model", "neural", "algorithm", "programming",
    "developer", "python", "rust", "database", "postgres", "kubernetes", "linux",
    "open-source", "open source", "security", "vulnerability", "cve", "hacker",
    "breach", "encryption", "chip", "semiconductor", "gpu", "data",
]

SKIP_KEYWORDS = [
    "promo code", "coupon", "deal", "discount", "% off", "sale", "sponsored",
    "best buy", "prime day", "recipe", "horoscope", "celebrity", "kardashian",
    "royal", "nfl", "nba", "soccer", "world cup", "fifa", "gift guide",
    "sweepstakes", "review:", "we tried", "connections hints", "wordle",
]


def label_title(title: str) -> str | None:
    """Return 'useful' | 'skipped' | None for a title."""
    t = f" {title.lower()} "
    if any(k in t for k in SKIP_KEYWORDS):
        return "skipped"
    if any(k in t for k in USEFUL_KEYWORDS):
        return "useful"
    return None
