"""Fetch adapters that turn remote sources into normalized article dicts."""
from __future__ import annotations

import datetime as dt
import logging
import time
from calendar import timegm

import feedparser
import httpx

from .sources import RssSource

logger = logging.getLogger(__name__)

USER_AGENT = "agentic-news-feed/0.1 (+https://github.com/v-shah07/agentic-news-feed)"
HN_API = "https://hacker-news.firebaseio.com/v0"


def _struct_time_to_dt(st) -> dt.datetime | None:
    if not st:
        return None
    try:
        return dt.datetime.fromtimestamp(timegm(st), tz=dt.timezone.utc)
    except Exception:
        return None


def fetch_hackernews(limit: int = 20, timeout: float = 20.0) -> list[dict]:
    """Fetch top HackerNews stories via the public Firebase API."""
    out: list[dict] = []
    with httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT}) as client:
        try:
            ids = client.get(f"{HN_API}/topstories.json").json()[:limit]
        except Exception as exc:  # network / parse failure
            logger.warning("HN topstories fetch failed: %s", exc)
            return out
        for hid in ids:
            try:
                item = client.get(f"{HN_API}/item/{hid}.json").json()
            except Exception as exc:
                logger.debug("HN item %s failed: %s", hid, exc)
                continue
            if not item or item.get("type") != "story" or not item.get("title"):
                continue
            url = item.get("url") or f"https://news.ycombinator.com/item?id={hid}"
            out.append(
                {
                    "source": "hackernews",
                    "source_type": "hn",
                    "external_id": str(hid),
                    "title": item.get("title", ""),
                    "content": item.get("text", "") or "",
                    "url": url,
                    "author": item.get("by"),
                    "published_at": dt.datetime.fromtimestamp(
                        item.get("time", 0), tz=dt.timezone.utc
                    )
                    if item.get("time")
                    else None,
                }
            )
    logger.info("HN fetched %d stories", len(out))
    return out


def fetch_rss(source: RssSource, limit: int = 20, timeout: float = 20.0) -> list[dict]:
    """Fetch and normalize an RSS/Atom feed (also used for Reddit RSS)."""
    out: list[dict] = []
    try:
        resp = httpx.get(
            source.url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("RSS fetch failed for %s: %s", source.name, exc)
        return out

    parsed = feedparser.parse(resp.content)
    for entry in parsed.entries[:limit]:
        link = entry.get("link")
        title = entry.get("title")
        if not link or not title:
            continue
        published = _struct_time_to_dt(
            entry.get("published_parsed") or entry.get("updated_parsed")
        )
        content = ""
        if entry.get("summary"):
            content = entry.get("summary")
        elif entry.get("content"):
            try:
                content = entry["content"][0].get("value", "")
            except Exception:
                content = ""
        author = entry.get("author")
        out.append(
            {
                "source": source.name,
                "source_type": source.source_type,
                "external_id": entry.get("id") or link,
                "title": title,
                "content": content,
                "url": link,
                "author": author,
                "published_at": published,
            }
        )
    logger.info("RSS %s fetched %d entries", source.name, len(out))
    # be polite to feed hosts
    time.sleep(0.1)
    return out
