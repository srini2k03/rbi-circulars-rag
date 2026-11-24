import { loadIndex } from "./index-loader.js";
import { makeTokenizer } from "./tokenizer.js";
import { BM25, denseScores, rrf, sigmoid, topK } from "./search.js";
import { buildUserPrompt, citedIds } from "./prompt.js";
import { generate, PROVIDERS } from "./llm.js";
import { headline, loadResults, renderDashboard } from "./dashboard.js";
import { advanced, docHtml, feedback, queryHistory, loadGraph, traceHtml, verificationHtml, verifyAnswer } from "./insight.js";

const params = new URLSearchParams(location.search);
const MODELS_ON = params.get("models") !== "off"; // ?models=off: BM25 only (tests, slow devices)
const INDEX_BASE = params.get("index") || "data/index";
const RESULTS_BASE = params.get("results") || "results";
const GRAPH_URL = params.get("graph") || "data/intel/graph.json";
const KEY_STORE = "rbirag.llmKey";
const STAGES = ["bm25", "dense", "rrf", "rerank", "gate"];

const $ = (sel) => document.querySelector(sel);
const state = { index: null, bm25: null, tokenize: null, last: null, seq: 0 };
const reducedMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

const ICON = {
  out: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 17 17 7M8 7h9v9"/></svg>`,
  alert: `<svg class="callout-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/></svg>`,
  key: `<svg class="callout-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="8" cy="15" r="4"/><path d="m10.8 12.2 9.2-9.2M17 6l3 3M15 8l2 2"/></svg>`,
  question: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6v.6M12 17h.01"/></svg>`,
  search: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>`,
};

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// Let the browser paint a state change before a long synchronous step.
const paint = () => new Promise((resolve) => {
  setTimeout(resolve, 50);
  requestAnimationFrame(() => setTimeout(resolve, 0));
});

const chipMs = (v) => (v < 1 ? "<1 ms" : `${Math.round(v).toLocaleString("en-US")} ms`);

function setStatus(text, kind = "") {
  const el = $("#status");
  el.textContent = text;
  el.className = `status ${kind}`;
}

function modelsActive() {
  const m = state.index?.manifest;
  return Boolean(MODELS_ON && m && !String(m.embedding.model || "").startsWith("fake") && state.index.emb.length);
}

// ---------- Pipeline stage chips ----------
function setStage(name, st, text = "") {
  const el = document.querySelector(`.stage[data-stage="${name}"]`);
  if (!el) return;
  el.dataset.state = st;
  el.querySelector(".stage__time").textContent = text;
}

function resetStages() {
  const on = modelsActive();
  for (const s of STAGES) {
    if (s === "bm25" || on) setStage(s, "idle");
    else setStage(s, "off", "off");
  }
}

// ---------- First-run model download meter ----------
function progressMeter(label, model) {
  const box = $("#model-progress");
  const files = new Map();
  let timer = null;
  const human = (b) => `${(b / 1048576).toFixed(1)} MB`;
  const update = () => {
    // The meter follows the model weights (>= 1 MB); config and tokenizer files are tiny and
    // would otherwise make the percentage jump back each time a new file starts.
    let loaded = 0, total = 0;
    for (const f of files.values()) if (f.total >= 1048576) { loaded += f.loaded; total += f.total; }
    const pct = total ? Math.min(100, Math.round((loaded / total) * 100)) : 0;
    $("#dl-label").textContent = `Downloading the ${label} (first search only; cached afterwards)`;
    $("#dl-pct").textContent = `${pct}%`;
    $("#dl-bar").style.width = `${pct}%`;
    box.querySelector('[role="progressbar"]').setAttribute("aria-valuenow", String(pct));
    $("#dl-detail").textContent = total ? `${model} · ${human(loaded)} of ${human(total)}` : `${model} · config and tokenizer`;
  };
  const onProgress = (ev) => {
    try {
      if (!ev || !ev.file) return;
      const f = files.get(ev.file) || { loaded: 0, total: 0 };
      if (ev.status === "progress" && ev.total) {
        f.total = ev.total;
        f.loaded = Math.min(ev.loaded || 0, ev.total);
      } else if (ev.status === "done") {
        f.loaded = f.total;
      }
      files.set(ev.file, f);
      update();
      // Only show the meter for a real download, not for a quick load from the cache.
      if (box.hidden && !timer) timer = setTimeout(() => { box.hidden = false; }, 250);
    } catch { /* the meter must never break model loading */ }
  };
  const finish = () => {
    clearTimeout(timer);
    box.hidden = true;
  };
  return { onProgress, finish };
}

class ModelError extends Error {
  constructor(cause) {
    super(cause?.message || String(cause));
    this.name = "ModelError";
  }
}

async function retrieve(question, live) {
  const { manifest: m, chunks, emb, scales, dim } = state.index;
  const stage = (...a) => { if (live()) setStage(...a); };
  const timings = {};
  stage("bm25", "run");
  await paint();
  let t = performance.now();
  const sparse = state.bm25.search(state.tokenize(question), m.retrieval.candidate_k);
  timings.bm25 = performance.now() - t;
  const trace = new Map();
  const tr = (i) => trace.get(i) || trace.set(i, {}).get(i);
  sparse.forEach((r, k) => Object.assign(tr(r.i), { bm25: r.score, bm25Rank: k + 1 }));
  stage("bm25", "done", chipMs(timings.bm25));
  let ranked = sparse.map((r) => ({ i: r.i, score: r.score }));
  let confidence = null;
  if (modelsActive()) {
    setStatus("Embedding the question (first run downloads the model) ...", "busy");
    stage("dense", "run", "loading model");
    let models;
    const dl1 = progressMeter("embedding model", m.embedding.browser_model);
    try {
      models = await import("./models.js");
      await models.loadEmbedder(m, dl1.onProgress);
    } catch (err) {
      stage("dense", "error", "model failed");
      throw new ModelError(err);
    } finally {
      dl1.finish();
    }
    stage("dense", "run");
    t = performance.now();
    const q = await models.embedQuery(question, m);
    const dense = topK(denseScores(q, emb, scales, dim), m.retrieval.candidate_k);
    timings.dense = performance.now() - t;
    dense.forEach((r, k) => Object.assign(tr(r.i), { dense: r.score, denseRank: k + 1 }));
    stage("dense", "done", chipMs(timings.dense));
    t = performance.now();
    const fused = rrf([sparse.map((r) => r.i), dense.map((r) => r.i)], m.retrieval.rrf_k);
    fused.forEach((r, k) => Object.assign(tr(r.id), { rrf: r.score, rrfRank: k + 1 }));
    const pool = fused.slice(0, m.retrieval.rerank_k);
    timings.rrf = performance.now() - t;
    stage("rrf", "done", chipMs(timings.rrf));
    setStatus("Reranking ...", "busy");
    stage("rerank", "run", "loading model");
    const dl2 = progressMeter("reranker", m.reranker.browser_model);
    try {
      await models.loadReranker(m, dl2.onProgress);
    } catch (err) {
      stage("rerank", "error", "model failed");
      throw new ModelError(err);
    } finally {
      dl2.finish();
    }
    stage("rerank", "run");
    t = performance.now();
    const scores = await models.rerank(question, pool.map((p) => chunks[p.id].text), m);
    timings.rerank = performance.now() - t;
    stage("rerank", "done", chipMs(timings.rerank));
    ranked = pool.map((p, j) => ({ i: p.id, score: scores[j] })).sort((a, b) => b.score - a.score || a.i - b.i);
    ranked.forEach((r, k) => Object.assign(tr(r.i), { rerank: r.score, rerankRank: k + 1 }));
    confidence = ranked.length ? sigmoid(ranked[0].score) : 0;
    const tau = m.retrieval.confidence_threshold;
    stage("gate", confidence < tau ? "low" : "pass", `${confidence.toFixed(2)} ${confidence < tau ? "<" : "≥"} τ`);
  }
  return { results: ranked.slice(0, m.retrieval.top_k).map((r) => ({ ...r, trace: trace.get(r.i) })), confidence, timings };
}

// ---------- Query-term highlighting ----------
const WORD = /[\p{L}\p{N}]+/gu;

function hitsOf(text, terms) {
  const hits = [];
  for (const mt of text.matchAll(WORD)) {
    const toks = state.tokenize(mt[0]);
    const hit = toks.find((tk) => terms.has(tk));
    if (hit) hits.push({ start: mt.index, end: mt.index + mt[0].length, term: hit });
  }
  return hits;
}

function highlight(text, terms) {
  let out = "", last = 0;
  for (const h of hitsOf(text, terms)) {
    out += escapeHtml(text.slice(last, h.start)) + `<mark>${escapeHtml(text.slice(h.start, h.end))}</mark>`;
    last = h.end;
  }
  return out + escapeHtml(text.slice(last));
}

// The ~320-character window that covers the most distinct query terms.
function snippet(text, terms, len = 320) {
  const flat = text.replace(/\s+/g, " ").trim();
  if (flat.length <= len) return highlight(flat, terms);
  const hits = hitsOf(flat, terms);
  let start = 0;
  if (hits.length) {
    let best = -1;
    for (const h of hits) {
      const s = Math.max(0, h.start - 60);
      const inWin = hits.filter((x) => x.start >= s && x.end <= s + len);
      const score = new Set(inWin.map((x) => x.term)).size * 100 + inWin.length;
      if (score > best) { best = score; start = s; }
    }
  }
  if (start > 0) {
    const sp = flat.indexOf(" ", start);
    start = sp >= 0 && sp < start + 30 ? sp + 1 : start;
  }
  let end = Math.min(flat.length, start + len);
  if (end < flat.length) {
    const sp = flat.lastIndexOf(" ", end);
    if (sp > start + len - 40) end = sp;
  }
  return `${start > 0 ? "… " : ""}${highlight(flat.slice(start, end), terms)}${end < flat.length ? " …" : ""}`;
}

function formatDate(d) {
  if (!d || !/^\d{4}-\d{2}-\d{2}/.test(d)) return escapeHtml(d || "");
  const dt = new Date(`${d.slice(0, 10)}T00:00:00Z`);
  if (isNaN(dt)) return escapeHtml(d);
  return dt.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
}

// ---------- Results ----------
function showPlaceholder(kind, detail = "") {
  const el = $("#results-empty");
  el.hidden = false;
  if (kind === "loading") {
    const sk = `<div class="skeleton-item sb-card" aria-hidden="true"><div class="sb-skeleton"></div><div class="sb-skeleton" style="width: 92%"></div><div class="sb-skeleton" style="width: 78%"></div></div>`;
    el.innerHTML = `<div class="loading-list" aria-busy="true"><p class="loading-label">Loading the index of RBI notifications ...</p>${sk}${sk}</div>`;
  } else if (kind === "error") {
    el.innerHTML = `<div class="sb-callout sb-callout--error" role="alert">${ICON.alert}<div class="callout-body">
      <p><strong>The search index could not be loaded.</strong></p><p>${escapeHtml(detail)}</p>
      <div class="callout-actions"><button type="button" class="sb-btn sb-btn--sm" data-reload>Reload the page</button></div></div></div>`;
    el.querySelector("[data-reload]").addEventListener("click", () => location.reload());
  } else {
    const m = state.index?.manifest;
    const n = m ? `${m.n_documents.toLocaleString()} notifications` : "the notifications";
    el.innerHTML = `<div class="empty-state"><span class="empty-state__icon">${ICON.search}</span>
      <h3>Ask a question to search ${escapeHtml(n)}</h3>
      <p>The top ${m?.retrieval?.top_k ?? 5} passages appear here, each with its notification, date, circular number,
      department and a link to the original on rbi.org.in. Query terms are highlighted.</p></div>`;
  }
}

function showSearching() {
  $("#results-empty").hidden = true;
  $("#results-head").hidden = false;
  $("#results-count").textContent = "";
  $("#search-meta").textContent = "searching ...";
  const list = $("#results");
  list.setAttribute("aria-busy", "true");
  const sk = (w) => `<li class="skeleton-item sb-card" aria-hidden="true"><div class="sb-skeleton"></div><div class="sb-skeleton" style="width: ${w}%"></div><div class="sb-skeleton" style="width: ${w - 14}%"></div></li>`;
  list.innerHTML = sk(94) + sk(88) + sk(91);
}

function showSearchError(err, question) {
  const list = $("#results");
  list.innerHTML = "";
  list.setAttribute("aria-busy", "false");
  $("#results-head").hidden = true;
  const el = $("#results-empty");
  el.hidden = false;
  const bm25Url = new URL(location.href);
  bm25Url.searchParams.set("models", "off");
  bm25Url.hash = "";
  const model = err instanceof ModelError;
  el.innerHTML = `<div class="sb-callout sb-callout--error" role="alert">${ICON.alert}<div class="callout-body">
    <p><strong>${model ? "The in-browser models could not be loaded." : "The search failed."}</strong></p>
    <p>${model ? "Dense retrieval and reranking download two small models from the Hugging Face hub (via cdn.jsdelivr.net) on the first search. " : ""}${escapeHtml(err.message)}</p>
    <div class="callout-actions"><button type="button" class="sb-btn sb-btn--sm" data-retry>Try again</button>
    ${model ? `<a class="sb-btn sb-btn--sm" href="${escapeHtml(bm25Url.href)}">Search with BM25 only (no download)</a>` : ""}</div></div></div>`;
  el.querySelector("[data-retry]").addEventListener("click", () => {
    $("#question").value = question;
    onSearch();
  });
}

function renderResults(question, res) {
  const { chunks, manifest: m } = state.index;
  const list = $("#results");
  list.innerHTML = "";
  list.setAttribute("aria-busy", "false");
  $("#results-empty").hidden = true;
  $("#results-head").hidden = false;
  $("#results-count").textContent = String(res.results.length);
  if (!res.results.length) {
    list.innerHTML = `<li class="empty">No matching passages. Try other words, or one of the example questions.</li>`;
    $("#search-meta").textContent = "";
    return;
  }
  const terms = new Set(state.tokenize(question));
  const gated = res.confidence != null;
  const top = res.results[0].score || 1;
  res.results.forEach((r, j) => {
    const c = chunks[r.i];
    const n = j + 1;
    const li = document.createElement("li");
    li.className = "result sb-card sb-spotlight";
    li.id = `src-${n}`;
    li.dataset.docId = c.doc_id;
    li.style.setProperty("--i", String(j));
    const rel = gated ? sigmoid(r.score) : Math.max(0, r.score / top);
    const scoreLabel = gated ? "relevance" : "BM25";
    const scoreVal = gated ? rel.toFixed(2) : r.score.toFixed(1);
    const tags = [
      c.date ? `<span class="sb-tag tag-date"><time datetime="${escapeHtml(c.date)}">${formatDate(c.date)}</time></span>` : "",
      c.circular_number ? `<span class="sb-tag">${escapeHtml(c.circular_number)}</span>` : "",
      c.department ? `<span class="sb-tag">${escapeHtml(c.department)}</span>` : "",
      c.section ? `<span class="sb-tag tag-section" title="${escapeHtml(c.section)}">§ ${escapeHtml(c.section)}</span>` : "",
    ].join("");
    li.innerHTML = `
      <div class="result-head"><span class="badge">[${n}]</span>
        <a class="result-title" href="${escapeHtml(c.url)}" target="_blank" rel="noopener">${escapeHtml(c.title || c.doc_id)}</a>
        <span class="sb-tag cited-tag" hidden>cited</span></div>
      ${tags ? `<div class="result-meta">${tags}</div>` : ""}
      <p class="snippet">${snippet(c.text, terms)}</p>
      <details class="passage-more"><summary>Full passage</summary><p class="passage">${highlight(c.text, terms)}</p></details>
      <div class="result-foot">
        <div class="score" title="${gated ? "sigmoid of the cross-encoder score" : "BM25 score (bar relative to the top hit)"}">
          <span>${scoreLabel}</span><span class="sb-meter" aria-hidden="true"><span style="width: ${(Math.min(1, rel) * 100).toFixed(0)}%"></span></span>
          <span class="score__val">${scoreVal}</span></div>
        <span class="result-actions"><button type="button" class="sb-btn sb-btn--sm sb-btn--ghost doc-btn" data-doc="${escapeHtml(c.doc_id)}" aria-haspopup="dialog">Document &amp; relations</button>
        <a class="sb-btn sb-btn--sm result-link" href="${escapeHtml(c.url)}" target="_blank" rel="noopener">rbi.org.in ${ICON.out}</a></span>
      </div>
      <div class="trace-wrap" data-advanced${advanced.get() ? "" : " hidden"}>${traceHtml(r.trace, n, m.retrieval.top_k)}</div>`;
    list.appendChild(li);
  });
  for (const b of list.querySelectorAll(".doc-btn")) b.addEventListener("click", () => openDoc(b.dataset.doc, b));
  const tau = m.retrieval.confidence_threshold;
  const conf = res.confidence == null ? "n/a (BM25 only)" : res.confidence.toFixed(3);
  const low = res.confidence != null && res.confidence < tau;
  $("#search-meta").innerHTML =
    `<span${low ? ` class="low"` : ""}>confidence ${escapeHtml(conf)}${low ? " (low: the answer step will say it doesn't know)" : ""}</span> · ` +
    Object.entries(res.timings).map(([k, v]) => `${k} ${v.toFixed(0)} ms`).join(" · ");
  $("#answer-btn").disabled = false;
  updateGateNote(res.confidence, tau);
}

function updateGateNote(confidence, tau) {
  const el = $("#gate-note");
  el.hidden = false;
  if (confidence == null) {
    el.dataset.kind = "off";
    el.textContent = "BM25 only: there is no reranker confidence in this mode, so the gate is not applied.";
  } else if (confidence < tau) {
    el.dataset.kind = "low";
    el.textContent = `Retrieval confidence ${confidence.toFixed(2)} is below τ = ${tau}: the answer step will say “I don't know” without calling the provider.`;
  } else {
    el.dataset.kind = "pass";
    el.textContent = `Retrieval confidence ${confidence.toFixed(2)} ≥ τ = ${tau}: the model answers from these passages.`;
  }
}

async function onSearch(ev) {
  ev?.preventDefault();
  const question = $("#question").value.trim();
  if (!question || !state.index) return;
  const seq = ++state.seq;
  const live = () => seq === state.seq;
  $("#answer").innerHTML = "";
  $("#gate-note").hidden = true;
  resetStages();
  showSearching();
  setStatus("Searching ...", "busy");
  try {
    const res = await retrieve(question, live);
    if (!live()) return;
    state.last = { question, ...res };
    renderResults(question, res);
    renderHistory(queryHistory.add(question));
    setStatus(`Done: ${res.results.length} passage${res.results.length === 1 ? "" : "s"}.`, "ok");
  } catch (err) {
    if (!live()) return;
    setStatus(`Search failed: ${err.message}`, "error");
    for (const s of STAGES) {
      const el = document.querySelector(`.stage[data-stage="${s}"]`);
      if (el?.dataset.state === "run") setStage(s, "error", "failed");
    }
    showSearchError(err, question);
  }
}

// ---------- Answers ----------
function linkify(answer, n) {
  return escapeHtml(answer).replace(/\[(\d{1,3})\]/g, (s, d) => (+d >= 1 && +d <= n ? `<a class="cite" href="#src-${d}" data-src="${d}">[${d}]</a>` : s));
}

function bindCitations() {
  for (const a of document.querySelectorAll("#answer .cite")) {
    const target = () => document.getElementById(`src-${a.dataset.src}`);
    const on = () => target()?.classList.add("is-hot");
    const off = () => target()?.classList.remove("is-hot");
    a.addEventListener("mouseenter", on);
    a.addEventListener("mouseleave", off);
    a.addEventListener("focus", on);
    a.addEventListener("blur", off);
  }
}

function markCited(ids) {
  for (const li of document.querySelectorAll("#results li.result")) {
    const cited = ids.includes(+li.id.replace("src-", ""));
    li.classList.toggle("is-cited", cited);
    const tag = li.querySelector(".cited-tag");
    if (tag) tag.hidden = !cited;
  }
}

function openKeySettings() {
  const d = $("#llm-settings");
  d.open = true;
  d.scrollIntoView({ block: "nearest", behavior: reducedMotion() ? "auto" : "smooth" });
  $("#api-key").focus({ preventScroll: true });
}

async function onAnswer() {
  const last = state.last;
  if (!last) return;
  const m = state.index.manifest;
  const provider = $("#provider").value;
  const label = PROVIDERS[provider].label;
  const key = $("#api-key").value.trim();
  if (!key) {
    $("#answer").innerHTML = `<div class="sb-callout sb-callout--info hint">${ICON.key}<div class="callout-body">
      <p class="hint-text">Add your own ${escapeHtml(label)} API key under “Answer generation” to generate an answer; the cited passages are shown below either way.</p>
      <p><button type="button" class="linklike" data-open-settings>Add a key</button></p></div></div>`;
    $("#answer [data-open-settings]").addEventListener("click", openKeySettings);
    return;
  }
  persistKey(provider, key);
  updateKeyState();
  if (last.confidence != null && last.confidence < m.retrieval.confidence_threshold) {
    $("#answer").innerHTML = `<div class="refusal" role="status"><span class="refusal__icon">${ICON.question}</span><div>
      <p class="refusal__text">${escapeHtml(m.prompt.refusal)}</p>
      <span class="meta">(retrieval confidence below the gate: ${last.confidence.toFixed(3)} &lt; τ = ${escapeHtml(m.retrieval.confidence_threshold)}, so nothing was sent to ${escapeHtml(label)})</span></div></div>`;
    return;
  }
  const passages = last.results.map((r) => state.index.chunks[r.i]);
  const seq = state.seq;
  setStatus(`Asking ${label} ...`, "busy");
  $("#answer").innerHTML = `<div class="answer-card sb-card answer-skeleton" aria-busy="true"><div class="answer-card__head">
    <span class="sb-label">Answer</span><span class="search-meta">asking ${escapeHtml(label)} ...</span></div>
    <div class="sb-skeleton" style="width: 96%"></div><div class="sb-skeleton" style="width: 88%"></div><div class="sb-skeleton" style="width: 64%"></div></div>`;
  $("#answer-btn").disabled = true;
  try {
    const out = await generate({
      provider,
      key,
      model: $("#model").value.trim() || m.generators[provider],
      system: m.prompt.system,
      user: buildUserPrompt(last.question, passages),
    });
    if (seq !== state.seq) return;
    const check = verifyAnswer(out.text, passages, state.tokenize);
    const shown = check.invalid.length
      ? out.text.replace(/\[\s*(\d{1,3}(?:\s*,\s*\d{1,3})*)\s*\]/g, (all, g) => {
        const keep = g.split(",").map((x) => x.trim()).filter((x) => !check.invalid.includes(Number(x)));
        return keep.length ? `[${keep.join(", ")}]` : "";
      })
      : out.text;
    const ids = citedIds(shown).filter((i) => i >= 1 && i <= passages.length);
    const refused = /^i don['’]t know/i.test(out.text.trim());
    state.lastAnswer = { question: last.question, answer: out.text, model: out.model, verdict: check.verdict,
      sources: passages.map((p) => p.doc_id) };
    $("#answer").innerHTML =
      `<article class="answer-card sb-card${refused ? " answer-card--refusal" : ""}">
        <div class="answer-card__head"><span class="sb-label">${refused ? "The model declined" : "Answer"}</span>
          <span class="sb-tag">${escapeHtml(out.model)}</span></div>
        <div class="answer-text">${linkify(shown, passages.length)}</div>
        ${refused ? "" : verificationHtml(check)}
        <p class="meta answer-card__foot">${escapeHtml(out.model)} | cites ${ids.length ? ids.map((i) => `[${i}]`).join(" ") : "nothing"} · answers can be wrong; check the cited notification on rbi.org.in</p>
        <div class="feedback" role="group" aria-label="Rate this answer">
          <span class="sb-label">Was this useful?</span>
          <button type="button" class="sb-btn sb-btn--sm sb-btn--ghost" data-rate="up" aria-label="Useful">Yes</button>
          <button type="button" class="sb-btn sb-btn--sm sb-btn--ghost" data-rate="down" aria-label="Not useful">No</button>
          <input class="sb-input feedback__note" type="text" maxlength="300" placeholder="Optional note (stays in this browser)" aria-label="Feedback note">
          <span class="feedback__state meta" role="status"></span>
        </div>
      </article>`;
    bindFeedback();
    bindCitations();
    markCited(ids);
    setStatus("Done.", "ok");
  } catch (err) {
    if (seq !== state.seq) return;
    setStatus(`Generation failed: ${err.message}`, "error");
    $("#answer").innerHTML = `<div class="sb-callout sb-callout--error" role="alert">${ICON.alert}<div class="callout-body">
      <p><strong>Generation failed.</strong> ${escapeHtml(err.message)}</p>
      <p>Check the key and the model name; the cited passages below are unaffected.</p></div></div>`;
  } finally {
    $("#answer-btn").disabled = false;
  }
}

// ---------- Key handling (memory by default; localStorage only when asked) ----------
function persistKey(provider, key) {
  try {
    if ($("#remember-key").checked) localStorage.setItem(KEY_STORE, JSON.stringify({ provider, key }));
    else localStorage.removeItem(KEY_STORE);
  } catch { /* storage may be unavailable */ }
}

function restoreKey() {
  try {
    const saved = JSON.parse(localStorage.getItem(KEY_STORE) || "null");
    if (saved?.key) {
      $("#provider").value = saved.provider;
      $("#api-key").value = saved.key;
      $("#remember-key").checked = true;
    }
  } catch { /* ignore */ }
}

function updateKeyState() {
  const el = $("#key-state");
  const has = Boolean($("#api-key").value.trim());
  el.dataset.state = has ? "set" : "none";
  el.textContent = !has ? "no key" : $("#remember-key").checked ? "key remembered" : "key in memory";
}

function updateProviderUI() {
  const provider = $("#provider").value;
  const p = PROVIDERS[provider];
  const link = $("#key-link");
  link.href = p.keyUrl;
  link.textContent = `Get a ${p.label} key`;
  const def = state.index?.manifest?.generators?.[provider];
  $("#model").placeholder = def ? `default: ${def}` : "default for the provider";
}

// ---------- Hero numbers (from the index manifest and results/) ----------
function setText(sel, text) {
  for (const el of document.querySelectorAll(sel)) el.textContent = text;
}

function fillManifestNumbers(m) {
  const years = !m.years ? "" : m.years[0] === m.years[1] ? String(m.years[0]) : `${m.years[0]}–${m.years[1]}`;
  setText('[data-stat="docs"]', m.n_documents.toLocaleString());
  setText('[data-stat="docs-ctx"]', `${m.n_chunks.toLocaleString()} passages${years ? ` · ${years}` : ""}`);
  setText('[data-funnel="docs"]', m.n_documents.toLocaleString());
  setText('[data-funnel="cand"]', String(m.retrieval.candidate_k));
  setText('[data-funnel="rerank"]', String(m.retrieval.rerank_k));
  setText('[data-funnel="top"]', String(m.retrieval.top_k));
  setText('[data-funnel="tau"]', String(m.retrieval.confidence_threshold));
}

async function fillResultNumbers() {
  let data;
  try {
    data = await loadResults(RESULTS_BASE);
  } catch {
    return; // keep the committed numbers in the markup
  }
  if (!data.meta) {
    for (const k of ["ndcg", "faith", "refusal"]) {
      setText(`[data-stat="${k}"]`, "–");
      setText(`[data-stat="${k}-ctx"]`, "evaluation in progress");
    }
    return;
  }
  const { best, gen } = headline(data);
  const two = (v) => (+v).toFixed(2);
  if (best) {
    const [lo, hi] = String(best["chunk_ndcg@10_ci"] || "").split(";");
    const n = best.n_queries ? ` · n = ${best.n_queries}` : "";
    const ci = lo && hi && !isNaN(+lo) && !isNaN(+hi) ? ` · 95% CI ${two(lo)}–${two(hi)}` : "";
    setText('[data-stat="ndcg"]', two(best["chunk_ndcg@10"]));
    setText('[data-stat="ndcg-ctx"]', `${best.config}${n}${ci}`);
  }
  if (gen) {
    const strip = (s) => String(s || "").replace(/ 100%$/, "");
    setText('[data-stat="faith"]', two(gen.judge_faithfulness));
    setText('[data-stat="faith-ctx"]', `${strip(gen.model)}, judged by ${strip(gen.judge_model)} · n = ${gen.n}`);
    if (gen.refusal_precision !== "" && !isNaN(+gen.refusal_precision)) {
      setText('[data-stat="refusal"]', two(gen.refusal_precision));
      const r = gen.refusal_recall !== "" && !isNaN(+gen.refusal_recall) ? ` · recall ${two(gen.refusal_recall)}` : "";
      setText('[data-stat="refusal-ctx"]', `“I don’t know” gate${r}`);
    }
  }
}

// ---------- Tabs ----------
let dashboard = null;

function showTab(name) {
  for (const t of ["search", "evaluation"]) {
    const on = t === name;
    $(`#tab-${t}`).hidden = !on;
    const b = $(`#tab-btn-${t}`);
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", String(on));
    b.tabIndex = on ? 0 : -1;
  }
  for (const a of document.querySelectorAll(".sb-nav__link[data-nav]")) {
    if (a.dataset.nav === name) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
  if (name === "evaluation" && !dashboard) {
    dashboard = renderDashboard($("#tab-evaluation"), RESULTS_BASE).catch((err) => {
      dashboard = null;
      $("#tab-evaluation").innerHTML = `<div class="sb-callout sb-callout--error eval-error" role="alert">${ICON.alert}<div class="callout-body">
        <h2>Could not render the evaluation results</h2><p>${escapeHtml(err.message)}</p></div></div>`;
    });
  }
}

function goTo(name, { focus = false } = {}) {
  showTab(name);
  $("#workspace").scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });
  if (focus && name === "search") $("#question").focus({ preventScroll: true });
}

function tabForHash(hash) {
  if (hash === "#evaluation") return "evaluation";
  if (hash === "" || hash === "#search" || hash.startsWith("#src-")) return "search";
  return null;
}

// ---------- Wiring ----------
$("#search-form").addEventListener("submit", onSearch);
$("#answer-btn").addEventListener("click", onAnswer);
$("#forget-key").addEventListener("click", () => {
  $("#api-key").value = "";
  $("#remember-key").checked = false;
  try { localStorage.removeItem(KEY_STORE); } catch { /* ignore */ }
  updateKeyState();
});
$("#api-key").addEventListener("input", updateKeyState);
$("#remember-key").addEventListener("change", updateKeyState);
$("#provider").addEventListener("change", updateProviderUI);
$("#tab-btn-search").addEventListener("click", () => {
  showTab("search");
  if (location.hash === "#evaluation") history.replaceState(null, "", location.pathname + location.search);
});
$("#tab-btn-evaluation").addEventListener("click", () => {
  showTab("evaluation");
  history.replaceState(null, "", "#evaluation");
});
$('[role="tablist"]').addEventListener("keydown", (e) => {
  if (e.key !== "ArrowLeft" && e.key !== "ArrowRight" && e.key !== "Home" && e.key !== "End") return;
  e.preventDefault();
  const next = $("#tab-btn-search").getAttribute("aria-selected") === "true" ? "evaluation" : "search";
  const target = e.key === "Home" ? "search" : e.key === "End" ? "evaluation" : next;
  $(`#tab-btn-${target}`).click();
  $(`#tab-btn-${target}`).focus();
});
for (const a of document.querySelectorAll("a[data-nav]")) {
  a.addEventListener("click", (e) => {
    e.preventDefault();
    const name = a.dataset.nav;
    history.replaceState(null, "", name === "evaluation" ? "#evaluation" : location.pathname + location.search);
    goTo(name, { focus: name === "search" });
  });
}
for (const el of document.querySelectorAll("[data-example]")) {
  el.addEventListener("click", () => { $("#question").value = el.textContent; onSearch(); });
}
document.addEventListener("keydown", (e) => {
  if (e.key !== "/" || e.metaKey || e.ctrlKey || e.altKey || e.isComposing) return;
  if (e.target.closest?.("input, textarea, select, [contenteditable]")) return;
  e.preventDefault();
  showTab("search");
  $("#question").focus();
});
if (location.hash === "#evaluation") showTab("evaluation");
window.addEventListener("hashchange", () => {
  const t = tabForHash(location.hash);
  if (t) showTab(t);
});

// ---------- History, feedback, advanced mode and the document panel ----------
function renderHistory(items = queryHistory.list()) {
  const box = $("#history");
  if (!box) return;
  box.hidden = !items.length;
  $("#history-list").innerHTML = items.map((q) => `<li><button type="button" class="chip chip--history">${escapeHtml(q)}</button></li>`).join("");
  for (const b of document.querySelectorAll(".chip--history")) {
    b.addEventListener("click", () => { $("#question").value = b.textContent; onSearch(); });
  }
}

function updateFeedbackCount() {
  const n = feedback.list().length;
  const el = $("#feedback-count");
  if (el) el.textContent = n ? `${n} rating${n === 1 ? "" : "s"} saved in this browser` : "No ratings yet";
  const btn = $("#feedback-export");
  if (btn) btn.disabled = !n;
}

function bindFeedback() {
  for (const b of document.querySelectorAll("#answer [data-rate]")) {
    b.addEventListener("click", () => {
      const note = $("#answer .feedback__note")?.value.trim() || "";
      feedback.add({ ...state.lastAnswer, rating: b.dataset.rate, note });
      for (const x of document.querySelectorAll("#answer [data-rate]")) x.setAttribute("aria-pressed", String(x === b));
      $("#answer .feedback__state").textContent = "Saved locally. Export it from the box above the results.";
      updateFeedbackCount();
    });
  }
}

async function openDoc(docId, opener) {
  const dlg = $("#doc-dialog");
  const body = $("#doc-body");
  $("#doc-title").textContent = state.index?.chunks.find((c) => c.doc_id === docId)?.title || docId;
  body.innerHTML = `<p class="meta"><span class="sb-spinner" aria-hidden="true"></span> Loading the relation graph ...</p>`;
  dlg.showModal?.() ?? dlg.setAttribute("open", "");
  dlg.dataset.opener = opener ? "1" : "";
  dlg._opener = opener;
  try {
    const g = await loadGraph(GRAPH_URL);
    body.innerHTML = docHtml(g, docId);
  } catch (err) {
    body.innerHTML = `<div class="sb-callout sb-callout--error" role="alert"><p>${escapeHtml(err.message)}</p></div>`;
  }
}

function initInsight() {
  const toggle = $("#advanced-toggle");
  if (toggle) {
    toggle.checked = advanced.get();
    toggle.addEventListener("change", () => {
      advanced.set(toggle.checked);
      for (const el of document.querySelectorAll("[data-advanced]")) el.hidden = !toggle.checked;
    });
  }
  $("#history-clear")?.addEventListener("click", () => { queryHistory.clear(); renderHistory([]); });
  $("#feedback-export")?.addEventListener("click", () => feedback.download());
  const dlg = $("#doc-dialog");
  dlg?.addEventListener("close", () => dlg._opener?.focus());
  $("#doc-close")?.addEventListener("click", () => (dlg.close ? dlg.close() : dlg.removeAttribute("open")));
  renderHistory();
  updateFeedbackCount();
}

async function init() {
  setStatus("Loading the index ...", "busy");
  showPlaceholder("loading");
  try {
    state.index = await loadIndex(INDEX_BASE);
  } catch (err) {
    setStatus(`Could not load the index: ${err.message}`, "error");
    for (const s of STAGES) setStage(s, "off", "");
    showPlaceholder("error", err.message);
    return;
  }
  const m = state.index.manifest;
  setStatus("Preparing BM25 ...", "busy");
  await paint();
  state.tokenize = makeTokenizer(m.bm25.stopwords);
  state.bm25 = new BM25(state.index.chunks.map((c) => state.tokenize(c.text)), m.bm25);
  const years = m.years ? `${m.years[0]}-${m.years[1]}` : "n/a";
  $("#index-info").textContent =
    `${m.n_chunks.toLocaleString()} passages from ${m.n_documents.toLocaleString()} RBI notifications (${years}); ` +
    `config ${m.config}${MODELS_ON && !String(m.embedding.model).startsWith("fake") ? "" : " (BM25 only: models disabled)"}.`;
  if (m.note) {
    $("#index-note").textContent = m.note;
    $("#index-note").hidden = false;
  }
  fillManifestNumbers(m);
  resetStages();
  showPlaceholder("ready");
  setStatus(modelsActive() ? "Ready. The first search downloads two small models; later searches reuse them." : "Ready.", "ok");
  $("#search-btn").disabled = false;
  restoreKey();
  updateKeyState();
  updateProviderUI();
}

updateProviderUI();
fillResultNumbers();
initInsight();
init();
