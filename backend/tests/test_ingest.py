"""Phase 1 tests: source catalog, normalization, dedupe, persistence."""
from __future__ import annotations

from backend.app.ingest import pipeline, run_ingest
from backend.app.ingest.sources import ALL_RSS_LIKE, source_count
from backend.app.models import Article


def test_source_count_is_at_least_ten():
    """The '10+ real-time sources' bullet must hold."""
    assert source_count() >= 10
    # HN + reddit + rss adapters all represented
    types = {s.source_type for s in ALL_RSS_LIKE}
    assert {"reddit", "rss"}.issubset(types)


def test_persist_dedupes_by_url(db_session):
    records = [
        {
            "source": "rss-a",
            "source_type": "rss",
            "title": "Same story",
            "content": "",
            "url": "https://example.com/story-1",
        },
        {
            "source": "rss-b",
            "source_type": "rss",
            "title": "Same story different feed",
            "content": "",
            "url": "https://example.com/story-1",  # duplicate URL
        },
        {
            "source": "rss-a",
            "source_type": "rss",
            "title": "Another",
            "content": "",
            "url": "https://example.com/story-2",
        },
    ]
    result = pipeline.IngestResult()
    pipeline._persist(db_session, records, result)
    db_session.commit()

    assert result.inserted == 2
    assert result.duplicates == 1
    assert db_session.query(Article).count() == 2


def test_run_ingest_uses_all_sources(monkeypatch, db_session):
    """run_ingest should hit HN + every RSS-like source without real network."""
    calls = {"hn": 0, "rss": []}

    def fake_hn(limit=20):
        calls["hn"] += 1
        return [
            {
                "source": "hackernews",
                "source_type": "hn",
                "title": "HN story",
                "content": "",
                "url": "https://news.ycombinator.com/item?id=1",
            }
        ]

    def fake_rss(src, limit=20):
        calls["rss"].append(src.name)
        return [
            {
                "source": src.name,
                "source_type": src.source_type,
                "title": f"{src.name} story",
                "content": "",
                "url": f"https://example.com/{src.name}",
            }
        ]

    monkeypatch.setattr(pipeline.adapters, "fetch_hackernews", fake_hn)
    monkeypatch.setattr(pipeline.adapters, "fetch_rss", fake_rss)

    result = run_ingest(db_session)

    assert calls["hn"] == 1
    assert len(calls["rss"]) == len(ALL_RSS_LIKE)
    assert result.sources_hit == source_count()
    assert result.inserted == source_count()
