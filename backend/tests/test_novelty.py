"""Phase 2 tests: novelty filter flags near-duplicates, keeps distinct stories."""
from __future__ import annotations

from backend.app.models import Article
from backend.app.novelty import filter_batch
from backend.tests.fakes import FakeEmbedder, InMemoryVectorStore


def _add(session, title, source):
    art = Article(source=source, source_type="rss", title=title, content="", url=f"u://{title}/{source}")
    session.add(art)
    session.flush()
    return art


def test_near_duplicates_flagged_redundant(db_session):
    # Three "ai chip" stories from different sources -> 1 novel, 2 redundant.
    _add(db_session, "New AI chip breaks records", "verge")
    _add(db_session, "AI chip sets new performance record", "techcrunch")
    _add(db_session, "The AI chip that broke records", "wired")
    # One clearly different story.
    _add(db_session, "Volcano erupts in Iceland", "bbc")
    db_session.commit()

    embedder = FakeEmbedder(
        {
            "ai chip": [1.0, 0.0, 0.0],
            "volcano": [0.0, 1.0, 0.0],
        }
    )
    store = InMemoryVectorStore()

    result = filter_batch(db_session, embedder, store, threshold=0.85, window_days=30)

    assert result.processed == 4
    assert result.redundant == 2  # 2 of the 3 ai-chip stories
    assert result.novel == 2      # first ai-chip + volcano
    assert 0 < result.reduction_pct < 100

    redundant = db_session.query(Article).filter(Article.is_redundant.is_(True)).count()
    assert redundant == 2


def test_all_articles_embedded_and_stored(db_session):
    _add(db_session, "Story one about ai chip", "a")
    _add(db_session, "Story two about volcano", "b")
    db_session.commit()

    embedder = FakeEmbedder({"ai chip": [1.0, 0.0, 0.0], "volcano": [0.0, 1.0, 0.0]})
    store = InMemoryVectorStore()
    filter_batch(db_session, embedder, store, threshold=0.85)

    assert store.count() == 2
    assert db_session.query(Article).filter(Article.embedded.is_(True)).count() == 2

    # Re-running processes nothing new (only_unembedded).
    again = filter_batch(db_session, embedder, store, threshold=0.85)
    assert again.processed == 0
