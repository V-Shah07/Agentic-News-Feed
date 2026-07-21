# Agentic Information Diet Manager — Build Spec

> **For the autonomous coding session.** Single source of truth for this repo. Build in phase order. Every phase ends with **PROVE IT**. The resume bullets at the bottom are the contract — each must be literally true and backed by a committed artifact (metrics log, MLflow run, test output) when you finish.

---

## 0. What you're building & the one risk to manage

An agentic pipeline that ingests news, filters redundant/irrelevant content via vector similarity + a learned interest profile, clusters and synthesizes daily digests, and exposes a RAG query interface over your reading history — orchestrated by LangChain agents calling tools.

**The one thing that makes this resume-differentiated is the PyTorch fine-tuning + MLflow model-registry story.** It's also the single biggest time risk in the whole 3-project sprint. So the strategy is: **get the entire pipeline working end-to-end on a base (un-fine-tuned) sentence-transformer FIRST**, prove every other bullet, and only THEN do a small-but-real fine-tuning run so that bullet is true. Never let fine-tuning block the rest of the pipeline.

### Scope decisions already made (do not re-expand)
- **CUT entirely:** LangGraph, Kafka, Prometheus, Grafana, Weights & Biases, AWS Lambda scheduling. MLflow alone covers the experiment-tracking + model-registry bullets — you do not also need W&B. The scheduler is a plain in-process loop / cron inside the container, not Lambda.
- **Fine-tuning is real but small.** A genuine fine-tune on your feedback data with logged before/after metrics. It does NOT need to be big or beat SOTA — it needs a true, measured improvement number vs. the base model on a held-out set.
- **Priority if time runs out:** provable bullets > polish. Cut the React frontend before cutting the fine-tuning metrics. A working pipeline + real MLflow runs beats a pretty dashboard.

---

## 1. Tech stack (final, simplified)

| Layer | Choice | Notes |
|---|---|---|
| Agents | LangChain (agents + custom tools) | no LangGraph |
| Embeddings | `sentence-transformers` | base model first, fine-tuned later |
| Fine-tuning | PyTorch + HF Transformers | small real run |
| Experiment tracking | **MLflow only** | experiment tracking + model registry |
| Vector store | ChromaDB | similarity + interest profile + RAG |
| LLM calls | OpenAI API | cluster summaries + RAG synthesis |
| Clustering | scikit-learn (k-means) | |
| Backend | FastAPI + PostgreSQL | raw articles + feedback |
| Task queue | Redis (simple) | keep light; a queue, not a saga |
| Scheduler | in-process loop / cron in container | NOT Lambda |
| Frontend | React + TypeScript | thin, last, sacrificial |
| Infra | Docker Compose | postgres + redis + chromadb + api |
| CI | GitHub Actions | build + test on push |

**Simplifications that are allowed and encouraged:**
- No FAISS *and* Chroma — Chroma alone is enough. Drop FAISS.
- No RabbitMQ — Redis for any queueing.
- Scheduler = APScheduler or a simple asyncio loop; running "every few hours" can be a configurable interval you can set to seconds for testing.
- Feedback signal can be seeded/simulated for the fine-tune (mark a labeled subset useful/skipped) so you're not blocked waiting for weeks of real reading history.

---

## 2. Build order (phases)

Each phase: build → **PROVE IT** → commit. **Measure and log reduction/quality metrics from the moment embeddings exist** — those logs become your bullet numbers.

### Phase 1 — Data flowing (target: first few hours)
- FastAPI skeleton + Postgres. Ingestion for HackerNews API + Reddit API + 3–5 RSS feeds.
- Raw articles saved with metadata: source, timestamp, title, content, URL.
- Docker Compose: `api`, `postgres`, `redis`. Add `chromadb` in Phase 2.
- Scheduler pulls on an interval (configurable; not Lambda).
- GitHub Actions CI from day one.

**PROVE IT:** `docker compose up`; scheduler pulls real articles from ≥10 sources total into Postgres; row count logged. (10+ sources backs the "10+ real-time sources" bullet — count individual RSS feeds + HN + Reddit.)

### Phase 2 — Embeddings + novelty filter ⭐ (headline bullet)
- Add ChromaDB. Embedding pipeline with a **base** `sentence-transformers` model.
- Each article embedded; query Chroma for semantic similarity vs last 30 days.
- Articles above a tunable similarity threshold flagged redundant.
- **Log raw count vs filtered count every run** — this is the dedup metric.

**PROVE IT:** run over a real ingested batch; log e.g. "ingested N, filtered M redundant (P% reduction)." Commit the log. That P% is the dedup bullet — read it off, don't invent it.

### Phase 3 — Relevance scoring + interest profile ⭐
- Maintain a user interest profile as a vector in Chroma.
- Score each article vs the profile by embedding similarity.
- Profile updates from implicit feedback (marked useful ↑, skipped ↓).

**PROVE IT:** feed a labeled set; show that "useful"-labeled articles score higher than "skipped" on average after profile update. Log the separation.

### Phase 4 — LangChain agent: cluster + synthesize ⭐ (headline bullet)
- LangChain agent with two custom tools:
  - `cluster_tool` — k-means over embeddings to group related stories.
  - `summarize_tool` — synthesize each cluster into one briefing via OpenAI.
- The agent decides dynamically whether to cluster vs summarize based on article volume (this "agent chooses tools" behavior is the agentic bullet — make the decision logic real, not hardcoded).

**PROVE IT:** run the agent on a day's articles; show it invoking both tools and producing a clustered digest. Log the tool-call trace. That trace proves "LangChain agents with custom tools."

### Phase 5 — RAG query interface ⭐
- All articles stored as embeddings in Chroma.
- FastAPI `/query` endpoint: natural language → semantic retrieval → LLM synthesis.
- "What happened in AI this week?" returns a reasoned answer across everything ingested.

**PROVE IT:** query returns a coherent answer citing retrieved articles. Commit an example request/response.

### Phase 6 — PyTorch fine-tuning + MLflow ⭐⭐ (the differentiator — do it LAST, do it REAL)
> Everything above works on the base model already. Now make the fine-tuning bullet true.
- Build training signal from feedback: "useful" vs "skipped" articles → contrastive/pair training examples for the sentence transformer.
- Fine-tune the base sentence-transformer in PyTorch on this data. **Small is fine.**
- **MLflow**: log hyperparameters, loss curves, and an embedding-quality / relevance metric on a **held-out set** per run. Run ≥2–3 configs so "tracked Y experiments across Z configs" is true.
- Register models in the **MLflow model registry**; promote the best version. The production embedding pipeline reads the promoted model.
- Compute a real **base vs fine-tuned improvement %** on the held-out relevance metric.

**PROVE IT:** MLflow UI shows ≥2 runs with logged params/metrics + a registered, promoted model. Log the base-vs-fine-tuned delta. That delta is the headline fine-tuning bullet — it must come from the held-out eval, not a guess.

### Phase 7 — Thin frontend + digest dashboard (sacrificial)
- Minimal React + TS: daily digest as ranked story clusters with relevance scores; a query box hitting `/query`. WebSocket live updates optional.
- **Cut entirely if short on time.** No bullet depends on it.

---

## 3. The fine-tuning de-risk plan (read this before Phase 6)

The failure mode is spending 2 days fighting fine-tuning and shipping nothing. Avoid it:
1. Pipeline must be fully working on the base model after Phase 5. If Phase 6 produced nothing, you still have a demo and 5 of 6 bullets.
2. Use a small base model (e.g. `all-MiniLM-L6-v2`) so training is minutes, not hours.
3. Seed/simulate the feedback labels if real ones are thin — a labeled subset is a legitimate training signal; be ready to say so honestly.
4. "Real but small" — the improvement % must be measured on a held-out set. Even a modest true improvement backs the bullet. A fabricated big number does not survive an interview.
5. Timebox it. If fine-tuning isn't converging in a couple hours, ship the base pipeline and log fine-tuning as "in progress."

---

## 4. Interview talking points (put in README)

- **Why fine-tune vs use the base model?** Personal reading history aligns embeddings to your interests; measured X% relevance lift on held-out feedback.
- **Why MLflow model registry?** Versioned experiments + automated promotion of the best model to the production embedding path; reproducible comparisons.
- **How does the agent decide to cluster vs summarize?** Tool selection is driven by article volume/state, not a fixed script — that's the "agentic" part.
- **How does novelty filtering work?** Cosine similarity in Chroma against a rolling 30-day window; tunable threshold trades recall vs dedup aggressiveness.
- **How does the interest profile learn?** Profile vector nudged by implicit feedback (useful ↑ / skipped ↓); relevance = similarity to that evolving vector.

---

## 5. Resume bullets (the contract — every one must end up TRUE)

Fill X/Y/Z from committed logs / the MLflow UI. Never invent numbers.

- Built agentic information pipeline using LangChain agents with custom tools for novelty detection, relevance scoring, and story synthesis across **10+ real-time sources** (count your RSS + HN + Reddit).
- Fine-tuned a PyTorch sentence transformer on personal reading history achieving **X% improvement in relevance vs the base model** on a held-out set (X = Phase 6 measured delta).
- Implemented an MLflow model registry with automated promotion — **tracked Y fine-tuning experiments across Z hyperparameter configs**, deploying the best embedding model to production (Y, Z = your actual runs).
- Implemented semantic deduplication via ChromaDB vector similarity achieving **X% reduction in redundant content vs raw feed** (X = Phase 2 log).
- Designed a personalized interest-modeling system with an implicit-feedback loop continuously updating user-profile embeddings.
- Deployed the ingestion pipeline via FastAPI + PostgreSQL + Redis, containerized with Docker Compose, on a scheduled ingestion loop.

> **Bullet honesty rule:** the fine-tuning and dedup numbers must be read off real logs. If Phase 6 gets cut, drop the two fine-tuning/MLflow bullets rather than fabricate them — the remaining bullets still make a strong project.
