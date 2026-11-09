// Loads the static index exported by `rbirag export-web`.

async function gunzipJSON(resp) {
  if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${resp.url}`);
  const stream = resp.body.pipeThrough(new DecompressionStream("gzip"));
  return JSON.parse(await new Response(stream).text());
}

async function buffer(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);
  return resp.arrayBuffer();
}

export async function loadIndex(base) {
  const manifest = await (await fetch(`${base}/manifest.json`)).json();
  const [chunkData, emb, scales] = await Promise.all([
    gunzipJSON(await fetch(`${base}/chunks.json.gz`)),
    buffer(`${base}/embeddings.i8`),
    buffer(`${base}/scales.f32`),
  ]);
  const f = chunkData.fields;
  const chunks = chunkData.rows.map((row) => Object.fromEntries(f.map((k, j) => [k, row[j]])));
  return {
    manifest,
    chunks,
    emb: new Int8Array(emb),
    scales: new Float32Array(scales),
    dim: manifest.embedding.dim,
  };
}
