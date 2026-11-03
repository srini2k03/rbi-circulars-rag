import { loadIndex } from "./index-loader.js";
import { makeTokenizer } from "./tokenizer.js";
import { BM25, denseScores, rrf, sigmoid, topK } from "./search.js";
import { buildUserPrompt, citedIds } from "./prompt.js";
import { generate, PROVIDERS } from "./llm.js";
import { renderDashboard } from "./dashboard.js";

const params = new URLSearchParams(location.search);
const MODELS_ON = params.get("models") !== "off"; // ?models=off: BM25 only (tests, slow devices)
const INDEX_BASE = params.get("index") || "data/index";
const KEY_STORE = "rbirag.llmKey";

const $ = (sel) => document.querySelector(sel);
const state = { index: null, bm25: null, tokenize: null, last: null };

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function setStatus(text, kind = "") {
  const el = $("#status");
  el.textContent = text;
  el.className = `status ${kind}`;
}

async function init() {
  setStatus("Loading the index ...");
  try {
    state.index = await loadIndex(INDEX_BASE);
  } catch (err) {
    setStatus(`Could not load the index: ${err.message}`, "error");
    return;
  }
  const m = state.index.manifest;
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
  setStatus("Ready.", "ok");
  $("#search-btn").disabled = false;
  restoreKey();
}

async function retrieve(question) {
  const { manifest: m, chunks, emb, scales, dim } = state.index;
  const timings = {};
  let t = performance.now();
  const sparse = state.bm25.search(state.tokenize(question), m.retrieval.candidate_k);
  timings.bm25 = performance.now() - t;
  let ranked = sparse.map((r) => ({ i: r.i, score: r.score }));
  let confidence = null;
  const realModel = !String(m.embedding.model || "").startsWith("fake");
  if (MODELS_ON && realModel && emb.length) {
    setStatus("Embedding the question (first run downloads the model) ...");
    t = performance.now();
    const { embedQuery, rerank } = await import("./models.js");
    const q = await embedQuery(question, m);
    const dense = topK(denseScores(q, emb, scales, dim), m.retrieval.candidate_k);
    timings.dense = performance.now() - t;
    const fused = rrf([sparse.map((r) => r.i), dense.map((r) => r.i)], m.retrieval.rrf_k);
    const pool = fused.slice(0, m.retrieval.rerank_k);
    setStatus("Reranking ...");
    t = performance.now();
    const scores = await rerank(question, pool.map((p) => chunks[p.id].text), m);
    timings.rerank = performance.now() - t;
    ranked = pool.map((p, j) => ({ i: p.id, score: scores[j] })).sort((a, b) => b.score - a.score || a.i - b.i);
    confidence = ranked.length ? sigmoid(ranked[0].score) : 0;
  }
  return { results: ranked.slice(0, m.retrieval.top_k), confidence, timings };
}

function renderResults(question, res) {
  const { chunks, manifest: m } = state.index;
  const list = $("#results");
  list.innerHTML = "";
  if (!res.results.length) {
    list.innerHTML = `<p class="empty">No matching passages.</p>`;
    return;
  }
  res.results.forEach((r, j) => {
    const c = chunks[r.i];
    const n = j + 1;
    const li = document.createElement("li");
    li.className = "result";
    li.id = `src-${n}`;
    li.dataset.docId = c.doc_id;
    li.innerHTML = `
      <div class="result-head"><span class="badge">[${n}]</span>
        <a href="${escapeHtml(c.url)}" target="_blank" rel="noopener">${escapeHtml(c.title || c.doc_id)}</a></div>
      <div class="meta">${escapeHtml([c.circular_number, c.date, c.department, c.section].filter(Boolean).join(" | "))}</div>
      <details><summary>${escapeHtml(c.text.slice(0, 220))}${c.text.length > 220 ? " ..." : ""}</summary>
        <p class="passage">${escapeHtml(c.text)}</p></details>`;
    list.appendChild(li);
  });
  const conf = res.confidence == null ? "n/a (BM25 only)" : res.confidence.toFixed(3);
  const low = res.confidence != null && res.confidence < m.retrieval.confidence_threshold;
  $("#search-meta").textContent =
    `confidence ${conf}${low ? " - low: the answer step will say it doesn't know" : ""} | ` +
    Object.entries(res.timings).map(([k, v]) => `${k} ${v.toFixed(0)} ms`).join(", ");
  $("#answer-btn").disabled = false;
}

async function onSearch(ev) {
  ev?.preventDefault();
  const question = $("#question").value.trim();
  if (!question || !state.index) return;
  $("#answer").innerHTML = "";
  setStatus("Searching ...");
  try {
    const res = await retrieve(question);
    state.last = { question, ...res };
    renderResults(question, res);
    setStatus("Done.", "ok");
  } catch (err) {
    setStatus(`Search failed: ${err.message}`, "error");
  }
}

function linkify(answer, n) {
  return escapeHtml(answer).replace(/\[(\d+)\]/g, (s, d) => (+d >= 1 && +d <= n ? `<a href="#src-${d}">[${d}]</a>` : s));
}

async function onAnswer() {
  const last = state.last;
  if (!last) return;
  const m = state.index.manifest;
  const provider = $("#provider").value;
  const key = $("#api-key").value.trim();
  if (!key) {
    $("#answer").innerHTML = `<p class="hint">Add your own ${PROVIDERS[provider].label} API key above to generate an answer; the cited passages are shown below either way.</p>`;
    return;
  }
  persistKey(provider, key);
  if (last.confidence != null && last.confidence < m.retrieval.confidence_threshold) {
    $("#answer").innerHTML = `<p class="refusal">${escapeHtml(m.prompt.refusal)} <span class="meta">(retrieval confidence below the gate)</span></p>`;
    return;
  }
  const passages = last.results.map((r) => state.index.chunks[r.i]);
  setStatus(`Asking ${PROVIDERS[provider].label} ...`);
  try {
    const out = await generate({
      provider,
      key,
      model: $("#model").value.trim() || m.generators[provider],
      system: m.prompt.system,
      user: buildUserPrompt(last.question, passages),
    });
    const ids = citedIds(out.text).filter((i) => i >= 1 && i <= passages.length);
    $("#answer").innerHTML =
      `<div class="answer-text">${linkify(out.text, passages.length)}</div>` +
      `<p class="meta">${escapeHtml(out.model)} | cites ${ids.length ? ids.map((i) => `[${i}]`).join(" ") : "nothing"}</p>`;
    setStatus("Done.", "ok");
  } catch (err) {
    setStatus(`Generation failed: ${err.message}`, "error");
  }
}

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

function showTab(name) {
  for (const t of ["search", "evaluation"]) {
    $(`#tab-${t}`).hidden = t !== name;
    $(`#tab-btn-${t}`).classList.toggle("active", t === name);
  }
  if (name === "evaluation") renderDashboard($("#tab-evaluation"), params.get("results") || "results");
}

$("#search-form").addEventListener("submit", onSearch);
$("#answer-btn").addEventListener("click", onAnswer);
$("#forget-key").addEventListener("click", () => {
  $("#api-key").value = "";
  $("#remember-key").checked = false;
  try { localStorage.removeItem(KEY_STORE); } catch { /* ignore */ }
});
$("#tab-btn-search").addEventListener("click", () => showTab("search"));
$("#tab-btn-evaluation").addEventListener("click", () => showTab("evaluation"));
for (const el of document.querySelectorAll("[data-example]")) {
  el.addEventListener("click", () => { $("#question").value = el.textContent; onSearch(); });
}
if (location.hash === "#evaluation") showTab("evaluation");
init();
