"""Phase 2 PROVE IT runner: embed ingested articles and log the dedup metric."""
from __future__ import annotations

import datetime as dt
import json
import logging
import os

from sqlalchemy import func, select

from backend.app.db import init_db, session_scope
from backend.app.embeddings import get_embedder
from backend.app.models import Article
from backend.app.novelty import filter_batch
from backend.app.vectorstore import get_articles_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("run_novelty")


def main() -> None:
    init_db()
    embedder = get_embedder()
    store = get_articles_store()

    with session_scope() as session:
        total = session.scalar(select(func.count(Article.id))) or 0
        result = filter_batch(session, embedder, store, only_unembedded=True)
        redundant_total = session.scalar(
            select(func.count(Article.id)).where(Article.is_redundant.is_(True))
        ) or 0

    payload = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "embedding_model": embedder.source,
        "is_finetuned": embedder.is_finetuned,
        "articles_total": total,
        "processed_this_run": result.processed,
        "redundant_this_run": result.redundant,
        "novel_this_run": result.novel,
        "reduction_pct_this_run": round(result.reduction_pct, 2),
        "redundant_total_in_db": redundant_total,
        "vector_count": store.count(),
    }

    print("\n===== PHASE 2 NOVELTY / DEDUP SUMMARY =====")
    print(json.dumps(payload, indent=2))
    print(
        f"\nINGESTED {result.processed}, FILTERED {result.redundant} REDUNDANT "
        f"({result.reduction_pct:.1f}% REDUCTION) vs raw feed"
    )

    os.makedirs("logs", exist_ok=True)
    with open("logs/phase2_novelty.log", "a") as fh:
        fh.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()
