// Retrieval primitives mirroring the Python pipeline: BM25 (rank_bm25's BM25Okapi),
// cosine over int8 embeddings, Reciprocal Rank Fusion and the sigmoid confidence.

export class BM25 {
  // docsTokens: array of token arrays (empty docs get a placeholder, as in Python)
  constructor(docsTokens, { k1 = 1.5, b = 0.75, epsilon = 0.25 } = {}) {
    this.k1 = k1; this.b = b;
    this.tf = [];
    this.len = new Float64Array(docsTokens.length);
    const df = new Map();
    let total = 0;
    docsTokens.forEach((toksIn, i) => {
      const toks = toksIn.length ? toksIn : ["\u0000"];
      const m = new Map();
      for (const t of toks) m.set(t, (m.get(t) || 0) + 1);
      this.tf.push(m);
      this.len[i] = toks.length;
      total += toks.length;
      for (const t of m.keys()) df.set(t, (df.get(t) || 0) + 1);
    });
    const n = docsTokens.length;
    this.avgdl = n ? total / n : 0;
    this.idf = new Map();
    let idfSum = 0;
    const negative = [];
    for (const [t, f] of df) {
      const v = Math.log(n - f + 0.5) - Math.log(f + 0.5);
      this.idf.set(t, v);
      idfSum += v;
      if (v < 0) negative.push(t);
    }
    const eps = epsilon * (this.idf.size ? idfSum / this.idf.size : 0);
    for (const t of negative) this.idf.set(t, eps);
  }

  scores(query) {
    const s = new Float64Array(this.tf.length);
    for (const q of query) {
      const idf = this.idf.get(q) || 0;
      if (!idf) continue;
      for (let i = 0; i < this.tf.length; i++) {
        const f = this.tf[i].get(q);
        if (!f) continue;
        s[i] += idf * (f * (this.k1 + 1)) /
          (f + this.k1 * (1 - this.b + (this.b * this.len[i]) / this.avgdl));
      }
    }
    return s;
  }

  // top-k [{i, score}] with score > 0, best first, ties by index
  search(query, k) {
    return topK(this.scores(query), k).filter((r) => r.score > 0);
  }
}

export function topK(scores, k) {
  const idx = Array.from(scores.keys());
  idx.sort((a, b) => scores[b] - scores[a] || a - b);
  return idx.slice(0, k).map((i) => ({ i, score: scores[i] }));
}

// Cosine similarity with int8 rows (vector ~= q * scale); query is L2-normalised.
export function denseScores(query, emb, scales, dim) {
  const n = scales.length;
  const out = new Float64Array(n);
  for (let r = 0; r < n; r++) {
    let dot = 0;
    const base = r * dim;
    for (let j = 0; j < dim; j++) dot += query[j] * emb[base + j];
    out[r] = dot * scales[r];
  }
  return out;
}

// Reciprocal Rank Fusion: score(d) = sum_i w_i / (k + rank_i(d)), ranks 1-based.
export function rrf(rankings, k = 60, weights = null) {
  const w = weights || rankings.map(() => 1);
  const score = new Map();
  const best = new Map();
  rankings.forEach((ranking, j) => {
    const seen = new Set();
    ranking.forEach((id, pos) => {
      if (seen.has(id)) return;
      seen.add(id);
      const rank = pos + 1;
      score.set(id, (score.get(id) || 0) + w[j] / (k + rank));
      best.set(id, Math.min(best.get(id) ?? Infinity, rank));
    });
  });
  return Array.from(score.entries())
    .map(([id, s]) => ({ id, score: s, best: best.get(id) }))
    .sort((a, b) => b.score - a.score || a.best - b.best || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

export function sigmoid(x) {
  return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
}
