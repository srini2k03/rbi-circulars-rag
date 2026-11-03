// Evaluation dashboard: renders results/ (tables + charts) when present.

function parseCSV(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(field); rows.push(row); row = []; field = "";
    } else field += ch;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  const [head, ...body] = rows.filter((r) => r.length > 1 || r[0]);
  return body.map((r) => Object.fromEntries(head.map((h, j) => [h, r[j]])));
}

const fmt = (v, d = 3) => (v === undefined || v === "" || isNaN(+v) ? "-" : (+v).toFixed(d));
const esc = (s) => String(s ?? "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);

function table(headers, rows) {
  return `<table><thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead>` +
    `<tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

async function getText(url) {
  const resp = await fetch(url);
  return resp.ok ? resp.text() : null;
}

export async function renderDashboard(root, base) {
  root.innerHTML = `<p class="status">Loading evaluation results ...</p>`;
  const metaText = await getText(`${base}/run_metadata.json`);
  if (!metaText) {
    root.innerHTML = `<div class="callout"><h2>Evaluation in progress</h2>
      <p>The retrieval ablation and generation evaluation run on GitHub Actions on the free tier
      and resume daily. Results appear here once the first report is committed.</p></div>`;
    return;
  }
  const meta = JSON.parse(metaText);
  const parts = [`<p class="test-set"><strong>Test set:</strong> ${esc(meta.test_set || "unknown")}</p>`,
    `<p class="meta">Generated ${esc(meta.generated_at)} at commit <code>${esc((meta.commit || "").slice(0, 7))}</code>.</p>`];
  if (meta.model_mix) {
    parts.push("<ul class='meta'>" + Object.entries(meta.model_mix).map(([k, v]) => `<li>${esc(k)}: ${esc(v)}</li>`).join("") + "</ul>");
  }
  const retr = await getText(`${base}/retrieval_ablation.csv`);
  if (retr) {
    const rows = parseCSV(retr).sort((a, b) => (+b["chunk_ndcg@10"] || 0) - (+a["chunk_ndcg@10"] || 0));
    parts.push("<h2>Retrieval ablation</h2>", table(
      ["config", "chunk R@1", "chunk R@5", "chunk R@10", "chunk MRR@10", "chunk nDCG@10", "doc R@5", "p50 ms"],
      rows.map((r) => [r.config, fmt(r["chunk_recall@1"]), fmt(r["chunk_recall@5"]), fmt(r["chunk_recall@10"]),
        fmt(r["chunk_mrr@10"]), fmt(r["chunk_ndcg@10"]), fmt(r["doc_recall@5"]), fmt(r.latency_ms_p50, 0)]),
    ));
  }
  const gen = await getText(`${base}/generation.csv`);
  if (gen) {
    const rows = parseCSV(gen);
    const measured = rows.filter((r) => (r.status || "measured") === "measured");
    const pending = rows.filter((r) => r.status && r.status !== "measured");
    parts.push("<h2>Generation</h2>");
    parts.push(measured.length ? table(
      ["config", "backend (model)", "n", "faithfulness (judge)", "faithfulness (NLI)", "relevance", "citation acc.", "refusal P", "refusal R"],
      measured.map((r) => [r.config, `${r.backend} (${r.model})`, r.n, fmt(r.judge_faithfulness), fmt(r.nli_faithfulness),
        fmt(r.judge_relevance), fmt(r.citation_accuracy), fmt(r.refusal_precision), fmt(r.refusal_recall)]),
    ) : "<p>No configuration is complete yet.</p>");
    if (pending.length) parts.push(`<p class="meta">Incomplete (not reported): ${pending.map((r) => esc(`${r.config}/${r.backend} ${r.status}`)).join("; ")}</p>`);
  }
  const charts = ["recall_at_k", "mrr_by_config", "chunk_size_sweep", "latency_vs_quality", "calibration"];
  const imgs = [];
  for (const c of charts) {
    const resp = await fetch(`${base}/charts/${c}.png`, { method: "HEAD" });
    if (resp.ok) imgs.push(`<figure><img src="${base}/charts/${c}.png" alt="${c}" loading="lazy"><figcaption>${c.replaceAll("_", " ")}</figcaption></figure>`);
  }
  if (imgs.length) parts.push("<h2>Charts</h2><div class='charts'>" + imgs.join("") + "</div>");
  root.innerHTML = parts.join("\n");
}

export { parseCSV };
