"""Phase 4 tests: clustering, and the agent's volume-driven tool selection."""
from __future__ import annotations

from backend.app.agent.digest import CLUSTER_VOLUME_THRESHOLD, DigestAgent
from backend.app.agent.tools import choose_k, cluster_embeddings
from backend.app.models import Article
from backend.tests.fakes import FakeEmbedder


class StubLLM:
    name = "stub"
    is_generative = False

    def summarize_cluster(self, titles, texts, max_words=90):
        return f"Briefing over {len(titles)} stories: {titles[0][:40]}"

    def answer(self, question, contexts, max_words=130):
        return "stub answer"


def test_choose_k_scales_with_volume():
    assert choose_k(1) == 1
    assert choose_k(3) == 1
    assert choose_k(12) == 2
    assert choose_k(60) == 10
    assert choose_k(1000) == 12  # capped


def test_cluster_embeddings_separates_two_groups():
    embs = [[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0.1, 0.9, 0]]
    labels = cluster_embeddings(embs, k=2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def _seed(session, titles):
    for i, t in enumerate(titles):
        session.add(Article(source="s", source_type="rss", title=t, content="",
                            url=f"u://{i}", is_redundant=False))
    session.commit()


def test_agent_clusters_when_volume_high(db_session):
    titles = [f"AI chip story {i}" for i in range(5)] + [f"volcano story {i}" for i in range(5)]
    _seed(db_session, titles)
    embedder = FakeEmbedder({"ai chip": [1.0, 0.0, 0.0], "volcano": [0.0, 1.0, 0.0]})
    agent = DigestAgent(llm=StubLLM(), embedder=embedder)

    digest = agent.run(db_session, since_hours=99999)

    assert digest.n_articles == 10
    assert digest.strategy == "cluster+summarize"
    tools_used = {t.tool for t in digest.trace}
    assert tools_used == {"cluster_tool", "summarize_tool"}  # BOTH invoked
    assert digest.n_clusters >= 2


def test_agent_skips_clustering_when_volume_low(db_session):
    _seed(db_session, ["AI chip lone story"])
    embedder = FakeEmbedder({"ai chip": [1.0, 0.0, 0.0]})
    agent = DigestAgent(llm=StubLLM(), embedder=embedder)

    digest = agent.run(db_session, since_hours=99999)

    assert digest.n_articles < CLUSTER_VOLUME_THRESHOLD
    assert digest.strategy == "summarize-only"
    tools_used = {t.tool for t in digest.trace}
    assert tools_used == {"summarize_tool"}  # cluster_tool NOT used
