// In-browser models via transformers.js (pinned), loaded lazily on first search.
const TRANSFORMERS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2";

let lib = null;
let extractor = null;
let reranker = null;

// A failed download is not cached, so the next search can try again.
function once(promise, reset) {
  return promise.catch((err) => { reset(); throw err; });
}

async function load() {
  if (!lib) lib = once(import(TRANSFORMERS), () => { lib = null; });
  return lib;
}

// `onProgress` receives transformers.js download events ({status, file, loaded, total});
// it is optional and only drives the download meter.
export function loadEmbedder(manifest, onProgress) {
  if (!extractor) {
    extractor = once((async () => {
      const { pipeline } = await load();
      return pipeline("feature-extraction", manifest.embedding.browser_model, {
        dtype: "q8",
        progress_callback: onProgress,
      });
    })(), () => { extractor = null; });
  }
  return extractor;
}

export function loadReranker(manifest, onProgress) {
  if (!reranker) {
    reranker = once((async () => {
      const { AutoTokenizer, AutoModelForSequenceClassification } = await load();
      const id = manifest.reranker.browser_model;
      const tok = await AutoTokenizer.from_pretrained(id, { progress_callback: onProgress });
      const model = await AutoModelForSequenceClassification.from_pretrained(id, {
        dtype: "q8",
        progress_callback: onProgress,
      });
      return { tok, model };
    })(), () => { reranker = null; });
  }
  return reranker;
}

export async function embedQuery(text, manifest) {
  const ex = await loadEmbedder(manifest);
  const out = await ex(manifest.embedding.query_instruction + text, { pooling: "cls", normalize: true });
  return out.data;
}

export async function rerank(query, texts, manifest) {
  const { tok, model } = await loadReranker(manifest);
  const inputs = tok(new Array(texts.length).fill(query), { text_pair: texts, padding: true, truncation: true });
  const { logits } = await model(inputs);
  return Array.from(logits.data);
}
