# Agentic Information Diet Manager

A news pipeline that ingests real-time content from a set of live sources, filters out redundant and irrelevant articles, groups and summarizes what is left into a daily digest, and answers natural-language questions over your reading history. The filtering runs on a `sentence-transformers` embedding model that is fine-tuned on your own implicit feedback (what you read versus what you skipped), with the experiments tracked and the best model promoted through an MLflow registry.

The point of the project is the ranking, not the reader. Most of the work is in learning an interest profile from feedback and in showing that fine-tuning the embedding model on that feedback measurably improves how well relevant articles rise to the top.

## Architecture

```
sources ─► ingestion ─► Postgres (raw articles)
 (HN,        (APScheduler       │
  Reddit,     in-process)       ▼
  RSS)                     embeddings ─► ChromaDB ──► novelty / dedup filter
                          (sentence-                  interest profile scoring
                           transformers)              RAG /query
                                                       │
                              LangChain agent ─────────┤ cluster_tool  (k-means)
                                                       └ summarize_tool
                                                              │
  PyTorch fine-tune ─► MLflow tracking + registry ─► promoted model
```

Sources are ingested on an in-process schedule into Postgres. Each article is embedded and stored in ChromaDB, where cosine similarity against a rolling window flags near-duplicate stories. An interest profile vector, learned from feedback, scores relevance. A LangChain agent decides whether the day's volume is high enough to cluster first, then summarizes. A RAG endpoint retrieves and cites articles for free-form questions.

## Key decisions

* **The novelty threshold is calibrated, not guessed.** The 0.70 cosine cutoff for "this is the same story" was read off the empirical similarity distribution rather than picked by hand, and every flagged cross-source pair was checked by eye (for example, one court ruling that surfaced across five outlets, an EU fine across two).
* **The agent's tool use is volume-driven.** Above a volume threshold it clusters with k-means and summarizes per cluster; below it, clustering is pointless overhead, so it summarizes directly. Both paths are exercised by tests.
* **Synthesis has no hard dependency on a paid API.** It runs on a local generative transformer (`distilbart-cnn`) by default and switches to OpenAI (`gpt-4o-mini`) when a key with quota is present. It is one code path, selected at runtime in `backend/app/llm.py`.
* **Fine-tuning is trained to match the metric it is judged on.** The contrastive pairs pull "useful" articles together and push "skipped" ones apart under cosine distance, which is the same distance the relevance ranking uses at serve time.

## Stack

FastAPI, PostgreSQL, Redis, ChromaDB, sentence-transformers, PyTorch, MLflow, LangChain, scikit-learn, OpenAI, Docker Compose, GitHub Actions.

## Quick start

```bash
cp .env.example .env            # add OPENAI_API_KEY to use the OpenAI synthesis path
docker compose up -d            # postgres + redis + chromadb + api
# API on http://localhost:8080  (docs at /docs)
```

Run a one-shot ingestion and print a summary:

```bash
python -m backend.run_ingest
```

## What the numbers look like

Everything below is reproducible from the committed 430-article snapshot (`backend/data/articles_snapshot.json`), with logs under `logs/`.

* **Ingestion.** One run pulled 398 articles from 19 live sources (HackerNews, four subreddits, fourteen RSS feeds).
* **Deduplication.** Embedding a 430-article corpus with the base `all-MiniLM-L6-v2` model flagged about 6% as redundant against a rolling 30-day window. Because this is a single-day cross-source snapshot, that figure is a conservative lower bound on the real rolling-window rate.
* **Interest profile.** Trained on a 70% split of a 220-article labeled subset and evaluated on the held-out 30%, mean relevance of "useful" articles came out to 0.274 versus 0.134 for "skipped" (a separation of 0.139, ranking AUC 0.872). The profile starts at zero separation and learns the split from feedback alone. It is live in the API through `POST /feedback` and `GET /articles?rank_by=relevance`.
* **Digest agent.** On a 120-article day the agent clustered into 12 groups (k-means) and summarized each, for 13 tool calls using both custom tools. The clusters were coherent (a CVE/security group, a promo-codes group, a malware group).
* **RAG.** `POST /query` embeds a question, retrieves the nearest articles from ChromaDB, and synthesizes a cited answer. A query about recent security vulnerabilities returned a grounded multi-sentence answer citing six retrieved articles at cosine 0.48 to 0.59.
* **Fine-tuning.** Fine-tuning the base model on feedback pairs and tracking the runs in MLflow, the best configuration (3 epochs, lr 5e-5, online-contrastive) lifted held-out AUC from 0.8945 to 0.9927, a 11.0% improvement. Held-out ranking error dropped from 0.106 to 0.007, and useful-versus-skipped separation rose from 0.205 to 0.325.

| run | loss | epochs | lr | held-out AUC | vs base |
|---|---|---|---|---|---|
| base-model | | | | 0.8945 | |
| online-cos-3ep-lr3e5 | online-contrastive | 3 | 3e-5 | 0.9764 | +9.2% |
| online-cos-4ep-lr3e5 | online-contrastive | 4 | 3e-5 | 0.9818 | +9.8% |
| online-cos-3ep-lr5e5 | online-contrastive | 3 | 5e-5 | 0.9927 | +11.0% |
| contrastive-cos-4ep-lr3e5 | contrastive | 4 | 3e-5 | 0.9509 | +6.3% |

The winning version is promoted through the MLflow model registry (the `production` alias). `backend/app/embeddings.py` loads the promoted model from `models/finetuned/production`, so the serving path picks it up with no code change. Browse the runs with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

There is also a thin React dashboard on top of the API. It is intentionally minimal; the backend is where the work is.

## Tests

```bash
pip install -r backend/requirements-ci.txt
pytest -q
```
