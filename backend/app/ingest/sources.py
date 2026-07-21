"""Declarative catalog of ingestion sources.

Counting sources for the "10+ real-time sources" bullet:
  - HackerNews API .................. 1
  - Reddit subreddits (RSS) ......... 3
  - Editorial RSS feeds ............. 7
  Total ............................. 11
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RssSource:
    name: str
    url: str
    source_type: str = "rss"


# HackerNews is handled by a dedicated adapter (JSON API), counted as one source.
HACKERNEWS_SOURCE_NAME = "hackernews"

REDDIT_SOURCES: list[RssSource] = [
    RssSource("reddit/r/technology", "https://www.reddit.com/r/technology/top/.rss?t=day", "reddit"),
    RssSource("reddit/r/programming", "https://www.reddit.com/r/programming/top/.rss?t=day", "reddit"),
    RssSource("reddit/r/MachineLearning", "https://www.reddit.com/r/MachineLearning/top/.rss?t=day", "reddit"),
]

RSS_SOURCES: list[RssSource] = [
    RssSource("techcrunch", "https://techcrunch.com/feed/"),
    RssSource("theverge", "https://www.theverge.com/rss/index.xml"),
    RssSource("arstechnica", "https://feeds.arstechnica.com/arstechnica/index"),
    RssSource("wired", "https://www.wired.com/feed/rss"),
    RssSource("bbc-tech", "https://feeds.bbci.co.uk/news/technology/rss.xml"),
    RssSource("engadget", "https://www.engadget.com/rss.xml"),
    RssSource("mit-tech-review", "https://www.technologyreview.com/feed/"),
]

ALL_RSS_LIKE: list[RssSource] = REDDIT_SOURCES + RSS_SOURCES


def source_count() -> int:
    """Total number of distinct real-time sources."""
    return 1 + len(ALL_RSS_LIKE)  # HN + reddit + rss
