// Reads a JSON spec from argv[2] and prints what the browser modules compute, so the
// Python test-suite can compare them with the Python implementation.
import { readFileSync } from "node:fs";
import { makeTokenizer } from "../js/tokenizer.js";
import { BM25, rrf, sigmoid, denseScores } from "../js/search.js";
import { buildUserPrompt, citedIds } from "../js/prompt.js";

const spec = JSON.parse(readFileSync(process.argv[2], "utf8"));
const tok = makeTokenizer(spec.stopwords);
const docsTokens = spec.docs.map((d) => tok(d));
const bm25 = new BM25(docsTokens, spec.bm25);
const out = {
  tokens: spec.texts.map((t) => tok(t)),
  scores: spec.queries.map((q) => Array.from(bm25.scores(tok(q)))),
  rrf: rrf(spec.rrf.rankings, spec.rrf.k).map((r) => [r.id, r.score]),
  sigmoid: spec.sigmoid.map(sigmoid),
  prompt: buildUserPrompt(spec.prompt.question, spec.prompt.chunks),
  cited: citedIds(spec.cited),
  dense: Array.from(denseScores(Float32Array.from(spec.dense.query), Int8Array.from(spec.dense.q), Float32Array.from(spec.dense.scales), spec.dense.dim)),
};
process.stdout.write(JSON.stringify(out));
