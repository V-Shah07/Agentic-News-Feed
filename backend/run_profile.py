"""Phase 3 PROVE IT runner.

Seed implicit feedback from a labeled subset, train the interest profile on a
train split, then show that on a HELD-OUT split 'useful' articles score higher
than 'skipped' ones. Logs the separation.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import random

from sqlalchemy import select

from backend.app.db import init_db, session_scope
from backend.app.embeddings import embedding_text, get_embedder
from backend.app.feedback_sim import label_title
from backend.app.models import Article, Feedback, UserProfile
from backend.app.profile import InterestProfile, save_profile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("run_profile")

random.seed(13)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    init_db()
    embedder = get_embedder()

    with session_scope() as session:
        # Reset any prior profile/feedback so the run is reproducible.
        session.query(Feedback).delete()
        session.query(UserProfile).delete()
        session.commit()

        articles = list(session.scalars(select(Article)))
        labeled = [(a, label_title(a.title)) for a in articles]
        labeled = [(a, lb) for a, lb in labeled if lb is not None]
        random.shuffle(labeled)

        # embed once
        emb_map = {
            a.id: embedder.encode_one(embedding_text(a.title, a.content))
            for a, _ in labeled
        }

        # 70/30 train/test split
        split = int(len(labeled) * 0.7)
        train, test = labeled[:split], labeled[split:]

        # --- baseline (no profile): separation is ~0 by construction ---
        empty = InterestProfile()
        base_useful = [empty.score(emb_map[a.id]) for a, lb in test if lb == "useful"]
        base_skip = [empty.score(emb_map[a.id]) for a, lb in test if lb == "skipped"]

        # --- train profile on the train split ---
        prof = InterestProfile()
        for a, lb in train:
            prof.update(emb_map[a.id], lb)
            session.add(Feedback(article_id=a.id, label=lb))
        save_profile(session, prof)
        session.commit()

        # --- evaluate on held-out test split ---
        test_useful = [prof.score(emb_map[a.id]) for a, lb in test if lb == "useful"]
        test_skip = [prof.score(emb_map[a.id]) for a, lb in test if lb == "skipped"]

        # rank-based metric: fraction of (useful, skipped) pairs correctly ordered (AUC)
        pairs = 0
        correct = 0
        for u in test_useful:
            for s in test_skip:
                pairs += 1
                if u > s:
                    correct += 1
                elif u == s:
                    correct += 0.5
        auc = correct / pairs if pairs else 0.0

    mu_useful, mu_skip = _mean(test_useful), _mean(test_skip)
    payload = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "embedding_model": embedder.source,
        "labeled_total": len(labeled),
        "train_size": len(train),
        "test_size": len(test),
        "test_useful_n": len(test_useful),
        "test_skipped_n": len(test_skip),
        "held_out_mean_relevance_useful": round(mu_useful, 4),
        "held_out_mean_relevance_skipped": round(mu_skip, 4),
        "separation": round(mu_useful - mu_skip, 4),
        "ranking_auc_useful_over_skipped": round(auc, 4),
        "baseline_separation_no_profile": round(_mean(base_useful) - _mean(base_skip), 4),
    }

    print("\n===== PHASE 3 INTEREST-PROFILE SEPARATION (held-out) =====")
    print(json.dumps(payload, indent=2))
    print(
        f"\nHELD-OUT: useful mean relevance {mu_useful:.3f} vs skipped {mu_skip:.3f} "
        f"(separation {mu_useful - mu_skip:+.3f}, ranking AUC {auc:.3f})"
    )

    os.makedirs("logs", exist_ok=True)
    with open("logs/phase3_profile.log", "a") as fh:
        fh.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()
