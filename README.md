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
| 3 | Interest profile + relevance scoring with implicit feedback | ✅ done |
| 4 | LangChain agent: `cluster_tool` + `summarize_tool` | ✅ done |
| 5 | RAG `/query` endpoint | ✅ done |
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

### Phase 3 evidence

`python -m backend.run_profile` seeds implicit feedback on a labeled subset (220
labeled articles), trains the interest-profile vector on a 70% split, and
evaluates on the **held-out 30%**: mean relevance of "useful" articles **0.274 vs
0.134** for "skipped" (separation **+0.139**, ranking **AUC 0.872**). The profile
starts at zero separation and learns the split purely from the feedback nudges.
Live in the API via `POST /feedback` → `GET /articles?rank_by=relevance`. See
`logs/phase3_profile.log`.

### Phase 4 evidence

`python -m backend.run_agent` runs a LangChain agent over the day's articles. The
agent's tool selection is **volume-driven**: with 120 articles (≥ threshold) it
invoked `cluster_tool` (k-means, k=12) → 12 clusters, then `summarize_tool` once
per cluster (13 tool calls, **both** custom tools used). Clusters came out
coherent — a cybersecurity/CVE group, a promo-codes group, a malware group. At
low volume the agent skips clustering and summarizes directly (unit-tested). See
`logs/phase4_agent_digest.log`.

> **LLM backend:** synthesis runs on a local generative transformer
> (`distilbart-cnn`) so it works with no external key; the backend auto-switches
> to OpenAI (`gpt-4o-mini`) when `OPENAI_API_KEY` has quota — one code path,
> selected at runtime in `backend/app/llm.py`.

### Phase 5 evidence

`POST /query` (or `python -m backend.run_rag`) embeds a natural-language question,
retrieves the nearest articles from ChromaDB, and synthesizes a cited answer. E.g.
*"What are the latest cybersecurity vulnerabilities and breaches?"* returned a
coherent multi-sentence answer grounded in 6 retrieved articles (WordPress RCE,
SonicWall 0-days, a CISA KEV entry, Firefox/Chrome patches) at cosine 0.48–0.59.
Full example request/response in `logs/phase5_rag_examples.log`.

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
