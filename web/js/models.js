// In-browser models via transformers.js (pinned), loaded lazily on first search.
const TRANSFORMERS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2";

let lib = null;
let extractor = null;
let rerankTok = null;
let rerankModel = null;

async function load() {
  if (!lib) lib = await import(TRANSFORMERS);
  return lib;
}

export async function embedQuery(text, manifest) {
  const { pipeline } = await load();
  if (!extractor) {
    extractor = await pipeline("feature-extraction", manifest.embedding.browser_model, { dtype: "q8" });
  }
  const out = await extractor(manifest.embedding.query_instruction + text, { pooling: "cls", normalize: true });
  return out.data;
}

export async function rerank(query, texts, manifest) {
  const { AutoTokenizer, AutoModelForSequenceClassification } = await load();
  if (!rerankModel) {
    const id = manifest.reranker.browser_model;
    rerankTok = await AutoTokenizer.from_pretrained(id);
    rerankModel = await AutoModelForSequenceClassification.from_pretrained(id, { dtype: "q8" });
  }
  const inputs = rerankTok(new Array(texts.length).fill(query), { text_pair: texts, padding: true, truncation: true });
  const { logits } = await rerankModel(inputs);
  return Array.from(logits.data);
}
