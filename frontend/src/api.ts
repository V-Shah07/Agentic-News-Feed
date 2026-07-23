// Thin API client. Requests go through the Vite dev proxy (/api -> :8080).
const BASE = import.meta.env.VITE_API_BASE ?? "/api";

export interface Cluster {
  cluster_id: number;
  size: number;
  briefing: string;
  top_titles: string[];
  article_ids: number[];
}

export interface Digest {
  generated_at: string;
  n_articles: number;
  strategy: string;
  n_clusters: number;
  llm_backend: string;
  clusters: Cluster[];
  trace: { tool: string; args: string }[];
}

export interface Source {
  article_id: number;
  similarity: number;
  title: string;
  source: string;
  url: string;
  snippet: string;
}

export interface QueryResult {
  query: string;
  answer: string;
  sources: Source[];
}

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const getDigest = (refresh = false) =>
  fetch(`${BASE}/digest?refresh=${refresh}`).then(j<Digest>);

export const askQuery = (query: string) =>
  fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, k: 6 }),
  }).then(j<QueryResult>);
