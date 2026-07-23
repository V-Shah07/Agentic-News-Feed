"""Phase 4 PROVE IT runner: run the digest agent and log the tool-call trace."""
from __future__ import annotations

import datetime as dt
import json
import logging
import os

from backend.app.agent.digest import DigestAgent
from backend.app.db import init_db, session_scope

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("run_agent")


def main() -> None:
    init_db()
    agent = DigestAgent()
    with session_scope() as session:
        digest = agent.run(session, since_hours=72)

    trace = [{"tool": t.tool, "args": t.args_summary, "output": t.output_summary}
             for t in digest.trace]
    briefings = [
        {
            "cluster_id": b.cluster_id,
            "size": b.size,
            "top_titles": b.top_titles,
            "briefing": b.briefing,
        }
        for b in digest.briefings
    ]
    payload = {
        "timestamp": digest.generated_at,
        "llm_backend": agent.llm.name,
        "llm_is_generative": getattr(agent.llm, "is_generative", False),
        "n_articles": digest.n_articles,
        "strategy": digest.strategy,
        "n_clusters": digest.n_clusters,
        "tool_calls": len(digest.trace),
        "distinct_tools_used": sorted({t.tool for t in digest.trace}),
        "trace": trace,
        "briefings": briefings,
    }

    print("\n===== PHASE 4 AGENT DIGEST =====")
    print(f"LLM backend: {payload['llm_backend']} (generative={payload['llm_is_generative']})")
    print(f"Strategy: {digest.strategy}  |  {digest.n_articles} articles  ->  "
          f"{digest.n_clusters} clusters  |  {len(digest.trace)} tool calls")
    print(f"Distinct tools invoked: {payload['distinct_tools_used']}")
    print("\n--- TOOL-CALL TRACE ---")
    for i, t in enumerate(trace, 1):
        print(f"  {i}. {t['tool']}({t['args']}) -> {t['output']}")
    print("\n--- CLUSTERED DIGEST (top clusters) ---")
    for b in briefings[:6]:
        print(f"\n  [cluster {b['cluster_id']} · {b['size']} stories]")
        print(f"    titles: {b['top_titles'][:2]}")
        print(f"    briefing: {b['briefing']}")

    os.makedirs("logs", exist_ok=True)
    with open("logs/phase4_agent_digest.log", "a") as fh:
        fh.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()
