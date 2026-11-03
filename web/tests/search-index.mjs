// node web/tests/search-index.mjs <base-url> <question> : load an exported index over HTTP
// with the browser loader and print the BM25 top hits as JSON.
import { loadIndex } from "../js/index-loader.js";
import { makeTokenizer } from "../js/tokenizer.js";
import { BM25 } from "../js/search.js";

const [base, question] = process.argv.slice(2);
const index = await loadIndex(base);
const tok = makeTokenizer(index.manifest.bm25.stopwords);
const bm25 = new BM25(index.chunks.map((c) => tok(c.text)), index.manifest.bm25);
const hits = bm25.search(tok(question), 3).map((r) => ({ doc_id: index.chunks[r.i].doc_id, score: r.score }));
process.stdout.write(JSON.stringify({ n: index.chunks.length, dim: index.dim, scales: index.scales.length, hits }));
