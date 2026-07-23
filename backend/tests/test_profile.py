"""Phase 3 tests: profile learning separates useful from skipped, persists."""
from __future__ import annotations

from backend.app.feedback_sim import label_title
from backend.app.models import Article
from backend.app.profile import (
    InterestProfile,
    apply_feedback,
    load_profile,
    save_profile,
)
from backend.tests.fakes import FakeEmbedder


def test_profile_moves_toward_useful_away_from_skipped():
    prof = InterestProfile()
    ai = [1.0, 0.0, 0.0]
    sports = [0.0, 1.0, 0.0]

    prof.update(ai, "useful")
    prof.update(ai, "useful")
    prof.update(sports, "skipped")

    # An unseen AI-like article should score higher than a sports-like one.
    assert prof.score([1.0, 0.0, 0.0]) > prof.score([0.0, 1.0, 0.0])
    assert prof.score([1.0, 0.0, 0.0]) > 0


def test_label_title_heuristic():
    assert label_title("New open-source LLM beats GPT on benchmarks") == "useful"
    assert label_title("Best Buy Prime Day deal: 40% off headphones") == "skipped"
    assert label_title("A quiet walk in the park") is None


def test_apply_feedback_persists_profile(db_session):
    art_ai = Article(source="a", source_type="rss", title="ai chip", content="", url="u://ai")
    art_sport = Article(source="b", source_type="rss", title="volcano", content="", url="u://v")
    db_session.add_all([art_ai, art_sport])
    db_session.commit()

    embedder = FakeEmbedder({"ai chip": [1.0, 0.0, 0.0], "volcano": [0.0, 1.0, 0.0]})

    apply_feedback(db_session, embedder, art_ai, "useful")
    apply_feedback(db_session, embedder, art_sport, "skipped")

    reloaded = load_profile(db_session)
    assert reloaded.dim == 3
    assert reloaded.n_useful == 1
    assert reloaded.n_skipped == 1
    assert reloaded.score([1.0, 0.0, 0.0]) > reloaded.score([0.0, 1.0, 0.0])
