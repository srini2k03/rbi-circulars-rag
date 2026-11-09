// BM25 tokenizer: a line-for-line port of rbirag.index.bm25.bm25_tokenize.
// NFKC + lower-case, ASCII alphanumeric runs, stopword removal, light plural folding.

export function stem(t) {
  if (t.length <= 3 || /^\d+$/.test(t)) return t;
  if (t.endsWith("ies") && t.length > 4) return t.slice(0, -3) + "y";
  if (t.endsWith("sses")) return t.slice(0, -2);
  if (t.endsWith("s") && !(t.endsWith("ss") || t.endsWith("us") || t.endsWith("is"))) {
    return t.slice(0, -1);
  }
  return t;
}

export function makeTokenizer(stopwords) {
  const sw = new Set(stopwords);
  return (text) => {
    const toks = (text.normalize("NFKC").toLowerCase().match(/[a-z0-9]+/g) || []);
    const out = [];
    for (const t of toks) if (!sw.has(t)) out.push(stem(t));
    return out;
  };
}
