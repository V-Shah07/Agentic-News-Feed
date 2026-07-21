# Agentic Information Diet Manager

An agentic news pipeline that ingests real-time content from 10+ sources, filters
redundant/irrelevant articles via vector similarity + a learned interest profile,
clusters and synthesizes daily digests with LangChain agents, and exposes a RAG
query interface over your reading history. Embeddings are powered by a
`sentence-transformers` model that is fine-tuned in PyTorch on implicit feedback,
with experiments tracked and the best model promoted through an MLflow registry.

## Architecture

```
sources ─► ingestion ─► Postgres (raw articles)
 (HN,        (APScheduler       │
  Reddit,     in-process)       ▼
  RSS)                     embeddings ─► ChromaDB ──► novelty filter (Phase 2)
                          (sentence-                  interest profile (Phase 3)
                           transformers)              RAG /query (Phase 5)
                                                       │
                              LangChain agent ─────────┤ cluster_tool  (k-means)
                              (Phase 4)                └ summarize_tool (OpenAI)

  PyTorch fine-tune (Phase 6) ─► MLflow tracking + registry ─► promoted model
```

## Stack

FastAPI · PostgreSQL · Redis · ChromaDB · sentence-transformers · PyTorch ·
MLflow · LangChain · scikit-learn · OpenAI · Docker Compose · GitHub Actions.

## Quick start

```bash
cp .env.example .env            # add OPENAI_API_KEY for Phase 4/5
docker compose up -d            # postgres + redis + chromadb + api
# API on http://localhost:8080  (docs at /docs)
```

Run a one-shot ingestion and print a summary:

```bash
python -m backend.run_ingest
```

## Build phases

| Phase | What | Status |
|---|---|---|
| 1 | FastAPI + Postgres + multi-source ingestion + scheduler + CI | ✅ done |
| 2 | ChromaDB embeddings + novelty/dedup filter | ✅ done |
| 3 | Interest profile + relevance scoring with implicit feedback | ⬜ |
| 4 | LangChain agent: `cluster_tool` + `summarize_tool` | ⬜ |
| 5 | RAG `/query` endpoint | ⬜ |
| 6 | PyTorch fine-tuning + MLflow registry | ⬜ |
| 7 | Thin React dashboard (sacrificial) | ⬜ |

### Phase 1 evidence

`python -m backend.run_ingest` pulled **398 articles from 19 live sources** in one
run (19 sources configured — HackerNews + 4 Reddit subreddits + 14 editorial RSS
feeds). See `logs/phase1_ingest.log`.

### Phase 2 evidence

`python -m backend.run_novelty` embedded a 430-article corpus with a base
`all-MiniLM-L6-v2` model and flagged **~6% as redundant** via ChromaDB cosine
similarity against a rolling 30-day window. The threshold (0.70) was **calibrated
from the empirical similarity distribution** — every flagged cross-source pair was
manually verified as the same story (the Paramount–Warner ruling appeared across 5
outlets, the AliExpress EU fine across 2, GPT-Red across 3). This single-day
cross-source snapshot is a conservative lower bound on the rolling-window rate. See
`logs/phase2_novelty.log` and `logs/phase2_threshold_calibration.log`.

## Interview talking points

- **Novelty filtering:** cosine similarity in Chroma against a rolling 30-day
  window; tunable threshold trades recall vs dedup aggressiveness.
- **Interest profile:** a profile vector nudged by implicit feedback (useful ↑ /
  skipped ↓); relevance = similarity to that evolving vector.
- **Agentic tool choice:** the agent picks `cluster` vs `summarize` from article
  volume/state, not a fixed script.
- **Why fine-tune:** aligning embeddings to personal reading history yields a
  measured relevance lift on held-out feedback (see Phase 6).
- **Why MLflow registry:** versioned experiments + automated promotion of the best
  model to the production embedding path; reproducible comparisons.

## Tests

```bash
pip install -r backend/requirements-ci.txt
pytest -q
```
