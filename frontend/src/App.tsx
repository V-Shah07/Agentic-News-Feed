import { useEffect, useState } from "react";
import { askQuery, getDigest, type Digest, type QueryResult } from "./api";

type Tab = "digest" | "ask";

export default function App() {
  const [tab, setTab] = useState<Tab>("digest");
  return (
    <div className="app">
      <header>
        <h1>Agentic Information Diet</h1>
        <nav>
          <button className={tab === "digest" ? "on" : ""} onClick={() => setTab("digest")}>
            Daily Digest
          </button>
          <button className={tab === "ask" ? "on" : ""} onClick={() => setTab("ask")}>
            Ask
          </button>
        </nav>
      </header>
      {tab === "digest" ? <DigestView /> : <AskView />}
    </div>
  );
}

function DigestView() {
  const [digest, setDigest] = useState<Digest | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = (refresh = false) => {
    setLoading(true);
    setError(null);
    getDigest(refresh)
      .then(setDigest)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  };

  useEffect(() => load(), []);

  if (loading) return <p className="muted">Running the agent…</p>;
  if (error) return <p className="error">Could not load digest: {error}</p>;
  if (!digest) return null;

  return (
    <section>
      <div className="meta">
        {digest.n_articles} articles → {digest.n_clusters} clusters ·{" "}
        <span className="tag">{digest.strategy}</span> ·{" "}
        <span className="tag">{digest.llm_backend}</span>{" "}
        <button className="link" onClick={() => load(true)}>
          refresh
        </button>
      </div>
      <div className="trace">
        agent tool calls: {digest.trace.map((t) => t.tool).join(" → ")}
      </div>
      {digest.clusters.map((c) => (
        <article key={c.cluster_id} className="card">
          <div className="card-head">
            <span className="badge">{c.size} stories</span>
          </div>
          <p className="briefing">{c.briefing}</p>
          <ul className="titles">
            {c.top_titles.map((t, i) => (
              <li key={i}>{t}</li>
            ))}
          </ul>
        </article>
      ))}
    </section>
  );
}

function AskView() {
  const [q, setQ] = useState("What happened in AI this week?");
  const [result, setResult] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!q.trim()) return;
    setLoading(true);
    setError(null);
    askQuery(q)
      .then(setResult)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  };

  return (
    <section>
      <form onSubmit={submit} className="ask-form">
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask about your reading history…" />
        <button type="submit" disabled={loading}>
          {loading ? "…" : "Ask"}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
      {result && (
        <div>
          <p className="answer">{result.answer}</p>
          <h3>Sources</h3>
          <ul className="sources">
            {result.sources.map((s) => (
              <li key={s.article_id}>
                <span className="sim">{s.similarity.toFixed(2)}</span>{" "}
                <a href={s.url} target="_blank" rel="noreferrer">
                  {s.title}
                </a>{" "}
                <span className="src">({s.source})</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
