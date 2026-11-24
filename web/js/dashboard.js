import { analysisCards } from "./analysis.js";
// Evaluation dashboard: renders results/ (heatmap, tables and charts) when present.
// Every number shown here is read from the committed results files at page load.

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
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const num = (v) => (v === undefined || v === null || v === "" || isNaN(+v) ? null : +v);
const uniq = (xs) => [...new Set(xs)];
const stripShare = (s) => String(s || "").replace(/ 100%$/, "");
// "data/eval/x.jsonl" in backticks -> <code>
const richText = (s) => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>");
const REPO = "https://github.com/srini2k03/rbi-circulars-rag";

// "complete" (formerly "measured"): at least 90% of the planned items answered and judged
const done = (r) => !r.status || r.status === "complete" || r.status === "measured";

function ci(v, d = 3) {
  const [lo, hi] = String(v || "").split(";").map(num);
  return lo == null || hi == null ? "" : `${lo.toFixed(d)}–${hi.toFixed(d)}`;
}

async function getText(url) {
  const resp = await fetch(url);
  return resp.ok ? resp.text() : null;
}

async function fetchResults(base) {
  const metaText = await getText(`${base}/run_metadata.json`);
  if (!metaText) return { meta: null, retrieval: null, generation: null };
  const meta = JSON.parse(metaText);
  const [retr, gen] = await Promise.all([getText(`${base}/retrieval_ablation.csv`), getText(`${base}/generation.csv`)]);
  return { meta, retrieval: retr ? parseCSV(retr) : null, generation: gen ? parseCSV(gen) : null };
}

const cache = new Map();
// Shared by the hero stats and the Evaluation tab, so results/ is fetched once.
export function loadResults(base) {
  if (!cache.has(base)) {
    const p = fetchResults(base);
    p.catch(() => cache.delete(base));
    cache.set(base, p);
  }
  return cache.get(base);
}

function bestRetrieval(rows) {
  return [...(rows || [])].filter((r) => num(r["chunk_ndcg@10"]) != null)
    .sort((a, b) => +b["chunk_ndcg@10"] - +a["chunk_ndcg@10"])[0] || null;
}

// Headline numbers: the best retrieval config by chunk nDCG@10, and the complete
// generation row with the largest sample (the full test set when it has run).
export function headline({ retrieval, generation }) {
  const gen = [...(generation || [])].filter((r) => done(r) && num(r.judge_faithfulness) != null)
    .sort((a, b) => (num(b.n) || 0) - (num(a.n) || 0))[0] || null;
  return { best: bestRetrieval(retrieval), gen };
}

function parseConfig(r) {
  const parts = String(r.config || "").split("-");
  return {
    chunker: r.chunker || parts[0],
    size: num(r.size) ?? num(parts[1]),
    mode: r.mode || parts.slice(2).join("-"),
  };
}

const MODE_LABEL = { bm25: "bm25", dense: "dense", hybrid: "hybrid", hybrid_rerank: "hybrid+rerank" };
const modeLabel = (m) => MODE_LABEL[m] || m;
const ordered = (known, vals) => [...known.filter((k) => vals.includes(k)), ...vals.filter((v) => !known.includes(v)).sort()];

function heatmapCard(rows) {
  const cells = rows.map((r) => ({ ...parseConfig(r), config: r.config, ndcg: num(r["chunk_ndcg@10"]), r5: num(r["doc_recall@5"]) }))
    .filter((c) => c.ndcg != null && c.chunker && c.size && c.mode);
  if (!cells.length) return "";
  const modes = ordered(["bm25", "dense", "hybrid", "hybrid_rerank"], uniq(cells.map((c) => c.mode)));
  const chunkers = ordered(["fixed", "semantic", "section"], uniq(cells.map((c) => c.chunker)));
  const sizes = uniq(cells.map((c) => c.size)).sort((a, b) => a - b);
  const values = cells.map((c) => c.ndcg);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const find = (m, c, s) => cells.find((d) => d.mode === m && d.chunker === c && d.size === s);
  const nq = num(rows[0]?.n_queries);
  const head = `<thead><tr><td></td>${chunkers.map((c) => `<th scope="colgroup" colspan="${sizes.length}">${esc(c)}</th>`).join("")}</tr>` +
    `<tr><td></td>${chunkers.map(() => sizes.map((s) => `<th scope="col">${s}</th>`).join("")).join("")}</tr></thead>`;
  const body = modes.map((m) => `<tr><th scope="row">${esc(modeLabel(m))}</th>${chunkers.map((c) => sizes.map((s) => {
    const d = find(m, c, s);
    if (!d) return `<td class="heat heat--na">–</td>`;
    const t = hi > lo ? (d.ndcg - lo) / (hi - lo) : 1;
    const best = d.ndcg === hi;
    const title = `${modeLabel(m)} · ${c} · ${s} tokens: chunk nDCG@10 ${d.ndcg.toFixed(3)}${d.r5 != null ? `, doc Recall@5 ${d.r5.toFixed(3)}` : ""}`;
    return `<td class="heat${best ? " is-best" : ""}" style="--p: ${Math.round(8 + t * 50)}%" title="${esc(title)}">${d.ndcg.toFixed(2)}</td>`;
  }).join("")).join("")}</tr>`).join("");
  const perMode = modes.map((m) => {
    const b = cells.filter((c) => c.mode === m).sort((x, y) => y.ndcg - x.ndcg)[0];
    return `${esc(modeLabel(m))} <strong>${b.ndcg.toFixed(3)}</strong> (${esc(b.chunker)}-${b.size})`;
  }).join(" · ");
  return `<figure class="ev-card sb-card">
    <figcaption class="ev-card__cap"><p class="sb-card__title">The ${cells.length}-configuration retrieval ablation</p>
      <p class="sb-card__sub">Chunk-level nDCG@10 on the test split${nq ? ` (${nq} answerable questions)` : ""}, by retrieval mode, chunker and chunk size in tokens. Darker green is better; the best cell is outlined.</p></figcaption>
    <div class="sb-table-wrap"><table class="heatmap"><caption class="sr-only">Chunk-level nDCG@10 for each retrieval mode, chunker and chunk size</caption>${head}<tbody>${body}</tbody></table></div>
    <div class="heat-legend" aria-hidden="true"><span>${lo.toFixed(2)}</span><span class="heat-legend__bar"></span><span>${hi.toFixed(2)}</span><span class="heat-legend__label">chunk nDCG@10</span></div>
    <p class="ev-note">Best per mode: ${perMode}.</p>
    <p class="sb-card__foot">Source: results/retrieval_ablation.csv</p>
  </figure>`;
}

const SHOWN = 10;

function rankedCard(rows) {
  const sorted = [...rows].sort((a, b) => (+b["chunk_ndcg@10"] || 0) - (+a["chunk_ndcg@10"] || 0));
  const top = num(sorted[0]?.["chunk_ndcg@10"]) || 1;
  const body = sorted.map((r, i) => {
    const v = num(r["chunk_ndcg@10"]);
    const bar = v == null ? "" : `<span class="bar" aria-hidden="true"><span style="width: ${Math.max(0, Math.min(100, (v / top) * 100)).toFixed(1)}%"></span></span>`;
    const c = ci(r["chunk_ndcg@10_ci"]);
    return `<tr${i === 0 ? ` class="is-best"` : ""}${i >= SHOWN ? " hidden data-extra" : ""}>
      <td class="cfg">${esc(r.config)}</td>
      <td class="num">${fmt(r["chunk_recall@1"])}</td><td class="num">${fmt(r["chunk_recall@5"])}</td>
      <td class="num">${fmt(r["chunk_recall@10"])}</td><td class="num">${fmt(r["chunk_mrr@10"])}</td>
      <td class="num"><span class="bar-cell">${bar}${fmt(r["chunk_ndcg@10"])}</span>${c ? `<span class="ci">${c}</span>` : ""}</td>
      <td class="num">${fmt(r["doc_recall@5"])}</td><td class="num">${fmt(r.latency_ms_p50, 0)}</td></tr>`;
  }).join("");
  const more = sorted.length > SHOWN
    ? `<div class="table-actions"><button type="button" class="sb-btn sb-btn--sm" data-show-all aria-expanded="false">Show all ${sorted.length} configurations</button></div>`
    : "";
  return `<figure class="ev-card sb-card">
    <figcaption class="ev-card__cap"><p class="sb-card__title">Retrieval ablation, ranked</p>
      <p class="sb-card__sub">Sorted by chunk nDCG@10, with its 95% bootstrap CI; p50 latency on a CPU runner.</p></figcaption>
    <div class="sb-table-wrap"><table class="sb-table rank-table">
      <thead><tr><th>config</th><th class="num">chunk R@1</th><th class="num">chunk R@5</th><th class="num">chunk R@10</th><th class="num">chunk MRR@10</th><th class="num">chunk nDCG@10</th><th class="num">doc R@5</th><th class="num">p50 ms</th></tr></thead>
      <tbody>${body}</tbody></table></div>
    ${more}
    <p class="sb-card__foot">Source: results/retrieval_ablation.csv</p>
  </figure>`;
}

function generationCard(rows, meta, headlineRow) {
  const measured = rows.filter(done);
  const pending = rows.filter((r) => !done(r));
  const metric = (r, k) => {
    const c = ci(r[`${k}_ci`]);
    return `<td class="num">${fmt(r[k])}${c ? `<span class="ci">${c}</span>` : ""}</td>`;
  };
  const judge = meta.judge?.description || stripShare(meta.judge_model);
  const table = measured.length ? `<div class="sb-table-wrap"><table class="sb-table gen-table">
      <thead><tr><th>generator · retrieval</th><th>questions</th><th>judge</th><th class="num">faithfulness<br>(judge)</th><th class="num">faithfulness<br>(NLI)</th><th class="num">relevance</th><th class="num">citation<br>acc.</th><th class="num">refusal<br>P</th><th class="num">refusal<br>R</th></tr></thead>
      <tbody>${measured.map((r) => `<tr${r === headlineRow ? ` class="is-best"` : ""}>
        <td><span class="gen-model">${esc(stripShare(r.model))}${r === headlineRow && measured.length > 1 ? `<span class="sb-tag headline-tag">headline</span>` : ""}</span>
          <span class="gen-sub">${esc(r.backend)} · ${esc(r.config)}</span></td>
        <td>${esc(r.sample || r.n)}</td>
        <td>${esc(stripShare(r.judge_model || "-"))}</td>
        ${metric(r, "judge_faithfulness")}${metric(r, "nli_faithfulness")}${metric(r, "judge_relevance")}
        ${metric(r, "citation_accuracy")}${metric(r, "refusal_precision")}${metric(r, "refusal_recall")}</tr>`).join("")}</tbody></table></div>`
    : `<p class="ev-note">No configuration is complete yet.</p>`;
  const pend = pending.length
    ? `<p class="ev-note">Incomplete (not reported): ${pending.map((r) => esc(`${r.config}/${r.backend} ${r.status}`)).join("; ")}</p>`
    : "";
  return `<figure class="ev-card sb-card">
    <figcaption class="ev-card__cap"><p class="sb-card__title">Answer quality</p>
      <p class="sb-card__sub">${judge ? `Answers judged by ${esc(judge)}; ` : ""}NLI faithfulness and citation metrics are judge-independent. 95% bootstrap CIs under each value.</p></figcaption>
    ${table}
    ${measured.length ? `<p class="ev-note">The judge and the NLI check can disagree sharply; both are reported. A row is complete when at least 90% of its planned questions were answered and judged.</p>` : ""}
    ${pend}
    <p class="sb-card__foot">Source: results/generation.csv</p>
  </figure>`;
}

function statsStrip(meta) {
  const s = meta.split;
  if (!s || meta.questions == null) return "";
  const ai = meta.ai_review?.reviewed_in_test_set;
  const tile = (v, label, ctx) => `<div class="sb-stat"><div class="sb-stat__value">${esc(v)}</div><div class="sb-stat__label">${esc(label)}</div><div class="sb-stat__ctx">${esc(ctx)}</div></div>`;
  return `<div class="sb-stats">
    ${tile(s.test_questions, "test questions", `${s.test_documents} documents; every reported number`)}
    ${tile(s.dev_questions, "dev questions", `${s.dev_documents} documents; config choice and gate only`)}
    ${tile(meta.questions, "Q/A pairs", `${meta.answerable} answerable · ${meta.unanswerable} unanswerable`)}
    ${tile(meta.human_verified ?? 0, "human-verified", ai != null ? `${ai} AI-reviewed (not human verification)` : "model-generated pairs")}
  </div>`;
}

function contextCards(meta) {
  const mix = meta.model_mix ? Object.entries(meta.model_mix) : [];
  const limits = Array.isArray(meta.limitations) ? meta.limitations : [];
  if (!mix.length && !limits.length && !meta.split) return "";
  const facts = [
    ...mix.map(([k, v]) => [k, v]),
    ...(meta.split?.method ? [["split", meta.split.method]] : []),
  ];
  const left = facts.length ? `<section class="sb-card ev-card"><p class="sb-card__title">Models that actually answered</p>
      <p class="sb-card__sub">Share of calls per role in the recorded run.</p>
      <dl class="facts" style="margin-top: 1rem">${facts.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl>
      <p class="sb-card__foot">Source: results/run_metadata.json</p></section>` : "";
  const right = limits.length ? `<section class="sb-card ev-card"><p class="sb-card__title">Limitations</p>
      <p class="sb-card__sub">As recorded with the run.</p>
      <ul class="limits" style="margin-top: 1rem">${limits.map((l) => `<li>${richText(l)}</li>`).join("")}</ul>
      <p class="sb-card__foot">Source: results/run_metadata.json</p></section>` : "";
  return `<div class="ev-grid">${left}${right}</div>`;
}

const CHARTS = [
  ["recall_at_k", "Recall@k, top configurations"],
  ["mrr_by_config", "MRR@10 by configuration", "tall"],
  ["latency_vs_quality", "Latency vs quality"],
  ["chunk_size_sweep", "Chunk-size sweep"],
  ["calibration", "Confidence-gate calibration"],
];

async function chartsGrid(base) {
  const found = await Promise.all(CHARTS.map(async ([c, label, shape]) => {
    try {
      const resp = await fetch(`${base}/charts/${c}.png`, { method: "HEAD" });
      return resp.ok ? [c, label, shape] : null;
    } catch { return null; }
  }));
  const imgs = found.filter(Boolean).map(([c, label, shape]) => `<figure class="chart-fig sb-card${shape ? ` chart-fig--${shape}` : ""}">
      <a class="chart-paper" href="${esc(`${base}/charts/${c}.png`)}" target="_blank" rel="noopener" aria-label="${esc(label)} (open full size)">
        <img src="${esc(`${base}/charts/${c}.png`)}" alt="${esc(c)}: ${esc(label)}" loading="lazy" decoding="async"></a>
      <figcaption>${esc(label)}<span>results/charts/${esc(c)}.png</span></figcaption></figure>`);
  return imgs.length
    ? `<section class="eval-charts"><p class="sb-kicker"><span class="sb-kicker__index">Charts</span><span class="sb-kicker__rule"></span>from rbirag eval report (matplotlib)</p>
      <div class="charts">${imgs.join("")}</div></section>`
    : "";
}

export async function renderDashboard(root, base) {
  root.setAttribute("aria-busy", "true");
  root.innerHTML = `<div class="eval-loading"><p class="status busy">Loading evaluation results ...</p>
    <div class="sb-skeleton" style="height: 2.6rem; width: min(100%, 30rem)"></div>
    <div class="sb-card"><div class="sb-skeleton" style="height: 13rem"></div></div></div>`;
  let data;
  try {
    data = await loadResults(base);
  } catch (err) {
    root.removeAttribute("aria-busy");
    root.innerHTML = `<div class="sb-callout sb-callout--error eval-error" role="alert"><div class="callout-body">
      <h2>Could not load the evaluation results</h2><p>${esc(err.message)}</p>
      <div class="callout-actions"><button type="button" class="sb-btn sb-btn--sm" data-retry>Try again</button></div></div></div>`;
    root.querySelector("[data-retry]").addEventListener("click", () => renderDashboard(root, base));
    return;
  }
  const { meta, retrieval, generation } = data;
  if (!meta) {
    root.removeAttribute("aria-busy");
    root.innerHTML = `<div class="sb-callout sb-callout--info eval-empty"><div class="callout-body"><h2>Evaluation in progress</h2>
      <p>The retrieval ablation and generation evaluation run on GitHub Actions on the free tier.
      Results appear here once the first report is committed.</p></div></div>`;
    return;
  }
  const commit = (meta.commit || "").slice(0, 7);
  const parts = [`<header class="eval-head">
      <p class="sb-kicker"><span class="sb-kicker__index">02</span><span class="sb-kicker__rule"></span>Evaluation</p>
      <h2 class="sb-h2">Measured on held-out <span class="sb-serif">notifications</span>.</h2>
      <p class="sb-lead">Every number on this tab is read from <code>results/</code> in the repository, which the evaluation
        workflow regenerates; nothing is typed in by hand.${meta.split ? " Reported numbers come from the test split: questions about notifications that were never used for tuning." : ""}</p>
    </header>`,
  `<div class="sb-callout sb-callout--warn test-set" role="note">
      <svg class="callout-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg>
      <div class="callout-body"><p><strong>Test set:</strong> ${richText(meta.test_set || "unknown")}</p>
      <p class="eval-meta">Generated ${esc(meta.generated_at)} at commit ${commit ? `<a href="${REPO}/commit/${esc(meta.commit)}"><code>${esc(commit)}</code></a>` : "<code>?</code>"}.</p></div>
    </div>`,
  statsStrip(meta)];
  if (retrieval) parts.push(heatmapCard(retrieval), rankedCard(retrieval));
  if (generation) parts.push(generationCard(generation, meta, headline(data).gen));
  parts.push(contextCards(meta));
  parts.push(await analysisCards(base));
  parts.push(await chartsGrid(base));
  root.removeAttribute("aria-busy");
  root.innerHTML = `<div class="eval">${parts.filter(Boolean).join("\n")}</div>`;
  const toggle = root.querySelector("[data-show-all]");
  toggle?.addEventListener("click", () => {
    const open = toggle.getAttribute("aria-expanded") !== "true";
    root.querySelectorAll("tr[data-extra]").forEach((tr) => { tr.hidden = !open; });
    toggle.setAttribute("aria-expanded", String(open));
    toggle.textContent = open ? `Show the top ${SHOWN}` : `Show all ${root.querySelectorAll("tr[data-extra]").length + SHOWN} configurations`;
  });
}

export { parseCSV };
