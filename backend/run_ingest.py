"""One-shot ingestion runner used for the Phase 1 PROVE IT evidence.

Usage:  python -m backend.run_ingest
Prints a summary line and per-source counts; also writes to logs/.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os

from sqlalchemy import func, select

from backend.app.db import init_db, session_scope
from backend.app.ingest import run_ingest
from backend.app.ingest.sources import source_count
from backend.app.models import Article

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("run_ingest")


def main() -> None:
    init_db()
    with session_scope() as session:
        before = session.scalar(select(func.count(Article.id))) or 0
        result = run_ingest(session)
        after = session.scalar(select(func.count(Article.id))) or 0
        by_source = dict(
            session.execute(
                select(Article.source, func.count(Article.id)).group_by(Article.source)
            ).all()
        )

    payload = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sources_configured": source_count(),
        "sources_hit": result.sources_hit,
        "fetched": result.fetched,
        "inserted": result.inserted,
        "duplicates": result.duplicates,
        "rows_before": before,
        "rows_after": after,
        "rows_by_source": by_source,
    }

    print("\n===== PHASE 1 INGEST SUMMARY =====")
    print(json.dumps(payload, indent=2))
    print(
        f"\nSOURCES CONFIGURED: {payload['sources_configured']}  "
        f"SOURCES HIT: {payload['sources_hit']}  "
        f"TOTAL ROWS IN POSTGRES: {after}"
    )

    os.makedirs("logs", exist_ok=True)
    with open("logs/phase1_ingest.log", "a") as fh:
        fh.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()
