"""Phase 5 tests: retrieval ranks relevant articles and answer cites sources."""
from __future__ import annotations

from backend.app.models import Article
from backend.app.rag import answer_query, retrieve
from backend.tests.fakes import FakeEmbedder, InMemoryVectorStore


class StubLLM:
    name = "stub"
    is_generative = False

    def answer(self, question, contexts, max_words=130):
        return f"Answer to {question!r} from {len(contexts)} sources: {contexts[0][:30]}"


def _seed(db_session, store, embedder):
    rows = [
        ("AI chip breaks performance records", "ai chip"),
        ("New volcano erupts in Iceland", "volcano"),
        ("Another AI accelerator announced", "ai chip"),
    ]
    for i, (title, _key) in enumerate(rows):
        art = Article(source=f"s{i}", source_type="rss", title=title, content="", url=f"u://{i}")
        db_session.add(art)
        db_session.flush()
        emb = embedder.encode_one(title)
        store.add([str(art.id)], [emb], [{"article_id": art.id, "source": art.source}], [title])
    db_session.commit()


def test_retrieve_ranks_relevant_first(db_session):
    embedder = FakeEmbedder({"ai chip": [1.0, 0.0, 0.0], "volcano": [0.0, 1.0, 0.0]})
    store = InMemoryVectorStore()
    _seed(db_session, store, embedder)

    hits = retrieve(db_session, embedder, store, "ai chip", k=3)
    assert hits[0].similarity >= hits[-1].similarity
    # top hit is an AI-chip article, not the volcano one
    assert "chip" in hits[0].title.lower() or "accelerator" in hits[0].title.lower()


def test_answer_query_cites_sources(db_session):
    embedder = FakeEmbedder({"ai chip": [1.0, 0.0, 0.0], "volcano": [0.0, 1.0, 0.0]})
    store = InMemoryVectorStore()
    _seed(db_session, store, embedder)

    ans = answer_query(db_session, embedder, store, StubLLM(), "ai chip", k=2)
    assert ans.answer
    assert len(ans.sources) == 2
    assert all(s.url for s in ans.sources)


def test_answer_query_handles_empty_store(db_session):
    embedder = FakeEmbedder({"ai chip": [1.0, 0.0, 0.0]})
    store = InMemoryVectorStore()
    ans = answer_query(db_session, embedder, store, StubLLM(), "anything", k=3)
    assert "No relevant" in ans.answer
    assert ans.sources == []
