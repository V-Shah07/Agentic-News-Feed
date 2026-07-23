"""Phase 5 PROVE IT runner: answer example queries with cited retrieval."""
from __future__ import annotations

import datetime as dt
import json
import logging
import os

from backend.app.db import init_db, session_scope
from backend.app.embeddings import get_embedder
from backend.app.llm import get_llm
from backend.app.rag import answer_query
from backend.app.vectorstore import get_articles_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("run_rag")

QUERIES = [
    "What happened in AI this week?",
    "What are the latest cybersecurity vulnerabilities and breaches?",
    "Any notable news about chips or semiconductors?",
]


def main() -> None:
    init_db()
    embedder = get_embedder()
    store = get_articles_store()
    llm = get_llm()

    results = []
    for q in QUERIES:
        with session_scope() as session:
            ans = answer_query(session, embedder, store, llm, q, k=6)
        results.append(ans)

    print("\n===== PHASE 5 RAG QUERY EXAMPLES =====")
    print(f"LLM backend: {llm.name}\n")
    payload_items = []
    for ans in results:
        print(f"Q: {ans.query}")
        print(f"A: {ans.answer}\n")
        print("   Sources:")
        for s in ans.sources:
            print(f"     [{s.similarity:.3f}] ({s.source}) {s.title[:70]}")
        print("-" * 70)
        payload_items.append(
            {
                "query": ans.query,
                "answer": ans.answer,
                "sources": [
                    {
                        "article_id": s.article_id,
                        "similarity": s.similarity,
                        "title": s.title,
                        "source": s.source,
                        "url": s.url,
                    }
                    for s in ans.sources
                ],
            }
        )

    payload = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "llm_backend": llm.name,
        "examples": payload_items,
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/phase5_rag_examples.log", "w") as fh:
        fh.write(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
