// Client-side companions of the Python pipeline: the lexical citation verifier
// (rbirag.generation.verify.LexicalSupport), query history and answer feedback (kept in
// this browser only), the advanced score trace and the document / relation panel.

const HISTORY_KEY = "rbirag.history";
const FEEDBACK_KEY = "rbirag.feedback";
const ADVANCED_KEY = "rbirag.advanced";

function store() {
  try { return window.localStorage; } catch { return null; }
}
function readJson(key, fallback) {
  try { return JSON.parse(store()?.getItem(key) || "null") ?? fallback; } catch { return fallback; }
}
function writeJson(key, value) {
  try { store()?.setItem(key, JSON.stringify(value)); return true; } catch { return false; }
}

export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// ---------- advanced mode ----------
export const advanced = {
  get: () => readJson(ADVANCED_KEY, false) === true,
  set: (on) => writeJson(ADVANCED_KEY, !!on),
};

// ---------- query history ----------
export const queryHistory = {
  list: () => readJson(HISTORY_KEY, []),
  add(q) {
    const items = [q, ...this.list().filter((x) => x !== q)].slice(0, 15);
    writeJson(HISTORY_KEY, items);
    return items;
  },
  clear: () => writeJson(HISTORY_KEY, []),
};

// ---------- feedback ----------
export const feedback = {
  list: () => readJson(FEEDBACK_KEY, []),
  add(record) {
    const items = [...this.list(), { ts: new Date().toISOString(), ...record }].slice(-500);
    writeJson(FEEDBACK_KEY, items);
    return items.length;
  },
  clear: () => writeJson(FEEDBACK_KEY, []),
  download() {
    const lines = this.list().map((r) => JSON.stringify(r)).join("\n") + "\n";
    const url = URL.createObjectURL(new Blob([lines], { type: "application/x-ndjson" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: "rbirag-feedback.jsonl" });
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  },
};

// ---------- citation verification (port of LexicalSupport) ----------
const NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
  "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"];
const NUMBER_MAP = Object.fromEntries([...NUMBER_WORDS.map((w, i) => [w, String(i)]),
  ["thirty", "30"], ["forty", "40"], ["fifty", "50"], ["sixty", "60"], ["ninety", "90"]]);

export function numbersIn(text) {
  const out = new Set((text.match(/\d+(?:[.,]\d+)*/g) || []).map((n) => n.replace(/,/g, "")));
  for (const w of text.toLowerCase().match(/\b[a-z]+\b/g) || []) if (NUMBER_MAP[w]) out.add(NUMBER_MAP[w]);
  return out;
}

const ABBREV = /(?:^|\s|\()(?:rs|no|nos|i\.e|e\.g|viz|para|paras|sr|dr|mr|mrs|ms|st|vol|art|sec|cl|ltd|pvt|co|inc|etc|vs|[a-z])\.$/i;

function sentences(text) {
  const out = [];
  for (const piece of text.split(/(?<=[.!?;])\s+(?=[A-Z0-9(₹])|\n+/)) {
    const p = piece.trim();
    if (!p) continue;
    if (out.length && ABBREV.test(out[out.length - 1])) out[out.length - 1] += ` ${p}`;
    else out.push(p);
  }
  return out;
}

function windows(text) {
  const s = sentences(text);
  if (s.length <= 3) return [s.join(" ")];
  const out = [];
  for (let i = 0; i < s.length - 2; i += 2) out.push(s.slice(i, i + 3).join(" "));
  out.push(s.slice(-3).join(" "));
  return out;
}

function splitClaims(sentence) {
  const parts = sentence.split(/;\s+|,\s+(?:and|but|while|whereas)\s+/).map((p) => p.trim().replace(/[.,]+$/, "")).filter(Boolean);
  const merged = [];
  for (const p of parts) {
    if (merged.length && p.split(/\s+/).length < 4) merged[merged.length - 1] += ` ${p}`;
    else merged.push(p);
  }
  return merged.length ? merged : [sentence];
}

export function verifyAnswer(answer, passages, tokenize, threshold = 0.6) {
  const n = passages.length;
  const wins = passages.map((p) => windows(p.text).map((w) => ({ w, toks: new Set(tokenize(w)), nums: numbersIn(w) })));
  const invalid = new Set();
  const claims = [];
  const body = answer.replace(/([.!?])((?:\s*\[\s*\d{1,3}(?:\s*[,\-–]\s*\d{1,3})*\s*\])+)/g, "$2$1");
  for (const sent of sentences(body)) {
    const cited = [];
    for (const m of sent.matchAll(/\[\s*(\d{1,3}(?:\s*,\s*\d{1,3})*)\s*\]/g)) {
      for (const x of m[1].split(",")) {
        const i = Number(x.trim());
        if (i >= 1 && i <= n) cited.push(i); else invalid.add(i);
      }
    }
    const text = sent.replace(/\[[^\]]*\]/g, "").trim();
    if (text.split(/\s+/).length < 3 || /^i don['’]t know/i.test(text)) continue;
    for (const claim of splitClaims(text)) {
      const ctoks = new Set(tokenize(claim));
      const cnums = numbersIn(claim);
      const score = (i) => Math.max(0, ...wins[i - 1].map(({ toks, nums }) => {
        for (const x of cnums) if (!nums.has(x)) return 0;
        let hit = 0;
        for (const t of ctoks) if (toks.has(t)) hit += 1;
        return ctoks.size ? hit / ctoks.size : 0;
      }));
      const pool = cited.length ? cited : passages.map((_, i) => i + 1);
      let best = 0; let src = null;
      for (const i of pool) { const s = score(i); if (s > best) { best = s; src = i; } }
      let status = best >= threshold ? "supported" : "unsupported";
      if (status === "unsupported" && cited.length) {
        for (let i = 1; i <= n; i++) {
          if (cited.includes(i)) continue;
          const s = score(i);
          if (s >= threshold && s > best) { best = s; src = i; status = "other-source"; }
        }
      }
      claims.push({ claim, cited, support: best, source: status === "unsupported" ? null : src, status });
    }
  }
  const ok = claims.filter((c) => c.status !== "unsupported").length;
  const verdict = !claims.length ? "no claims" : ok === claims.length ? "supported" : ok ? "partially supported" : "unsupported";
  return { claims, verdict, invalid: [...invalid].sort((a, b) => a - b) };
}

export function verificationHtml(v) {
  if (!v.claims.length && !v.invalid.length) return "";
  const label = { supported: "supported by the cited passage", "other-source": "supported by another retrieved passage", unsupported: "not found in the retrieved passages" };
  const rows = v.claims.map((c) => `<li class="claim claim--${c.status}"><span class="claim__mark" aria-hidden="true">${c.status === "unsupported" ? "!" : "✓"}</span>
      <span class="claim__text">${esc(c.claim)}</span>
      <span class="claim__meta">${label[c.status]}${c.source ? ` [${c.source}]` : ""} · overlap ${c.support.toFixed(2)}</span></li>`).join("");
  const warn = v.invalid.length ? `<p class="claim__warn">Cited ${v.invalid.map((i) => `[${i}]`).join(" ")}, which ${v.invalid.length > 1 ? "do" : "does"} not exist: removed.</p>` : "";
  const note = v.verdict === "unsupported" ? `<p class="claim__warn">The retrieved documents do not establish this answer.</p>` : "";
  return `<details class="verify" ${v.verdict !== "supported" ? "open" : ""}>
    <summary><span class="sb-label">Citation check</span> <span class="sb-tag verify__verdict verify__verdict--${v.verdict.replace(/\s/g, "-")}">${esc(v.verdict)}</span></summary>
    <p class="verify__def">Each claim is checked against short windows of its cited passage: every figure must appear there and most of its words must match. A lexical check, not a proof: it can miss paraphrases and cannot judge legal meaning.</p>
    ${note}${warn}<ol class="claims">${rows}</ol></details>`;
}

// ---------- advanced score trace ----------
export function traceHtml(t, finalRank, contextSize) {
  if (!t) return "";
  const cell = (score, rank, digits) => (score == null ? `<td class="na">-</td>` : `<td>${Number(score).toFixed(digits)}${rank ? ` <span class="rank">#${rank}</span>` : ""}</td>`);
  return `<table class="trace"><caption class="sr-only">Stage scores of this passage</caption>
    <thead><tr><th scope="col">BM25</th><th scope="col">dense cosine</th><th scope="col">fusion</th><th scope="col">cross-encoder</th><th scope="col">final</th></tr></thead>
    <tbody><tr>${cell(t.bm25, t.bm25Rank, 2)}${cell(t.dense, t.denseRank, 3)}${cell(t.rrf, t.rrfRank, 4)}${cell(t.rerank, t.rerankRank, 2)}
    <td>#${finalRank}${finalRank <= contextSize ? ` <span class="sb-tag in-context">in context</span>` : ""}</td></tr></tbody></table>`;
}

// ---------- document panel ----------
let graphPromise = null;
export function loadGraph(url) {
  if (!graphPromise) {
    graphPromise = fetch(url).then((r) => {
      if (!r.ok) throw new Error(`relation graph unavailable (${r.status})`);
      return r.json();
    }).then((g) => {
      const out = new Map(); const inc = new Map();
      for (const [a, b, type, conf, ev] of g.edges) {
        (out.get(a) || out.set(a, []).get(a)).push({ doc: b, type, conf, ev });
        (inc.get(b) || inc.set(b, []).get(b)).push({ doc: a, type, conf, ev });
      }
      const famOf = new Map();
      for (const ids of Object.values(g.families || {})) for (const id of ids) famOf.set(id, ids);
      return { ...g, out, inc, famOf };
    }).catch((e) => { graphPromise = null; throw e; });
  }
  return graphPromise;
}

const CATEGORY = { master_direction: "Master Direction", master_circular: "Master Circular", directions: "Directions",
  regulations: "Regulations", ap_dir_circular: "A.P. (DIR Series) circular", notification: "Notification", circular: "Circular" };
const CHANGE = new Set(["amends", "supersedes", "partially_modifies"]);

export function docHtml(g, docId) {
  const d = g.docs[docId];
  if (!d) return `<p>No metadata for ${esc(docId)}.</p>`;
  const title = (id) => esc(g.docs[id]?.t || id);
  const date = (id) => esc(g.docs[id]?.d || "undated");
  const versions = (g.famOf.get(docId) || [docId]).slice().sort((a, b) => (g.docs[a]?.d || "").localeCompare(g.docs[b]?.d || ""));
  const changedBy = (g.inc.get(docId) || []).filter((e) => CHANGE.has(e.type));
  const changes = (g.out.get(docId) || []).filter((e) => CHANGE.has(e.type));
  const refs = (g.out.get(docId) || []).filter((e) => e.type === "refers_to");
  const citedBy = (g.inc.get(docId) || []).filter((e) => e.type === "refers_to");
  const newer = versions.filter((v) => (g.docs[v]?.d || "") > (d.d || ""));
  const superseders = changedBy.filter((e) => e.type === "supersedes").map((e) => e.doc);
  const latest = [...newer, ...superseders].sort((a, b) => (g.docs[b]?.d || "").localeCompare(g.docs[a]?.d || ""))[0];
  const item = (e) => `<li><span class="sb-tag">${esc(e.type.replace("_", " "))}</span> ${title(e.doc)} <span class="meta">(${date(e.doc)}, ${esc(e.conf)} confidence)</span></li>`;
  const eff = d.ek === "immediate" ? "with immediate effect" : d.ek === "gazette" ? "from publication in the Official Gazette" : d.e || "not stated (or not found)";
  return `<dl class="docmeta">
      <dt>Category</dt><dd>${esc(CATEGORY[d.c] || d.c)}</dd>
      <dt>Published</dt><dd>${esc(d.d || "unknown")}</dd>
      <dt>Effective</dt><dd>${esc(eff)}</dd>
      ${d.n ? `<dt>Circular no.</dt><dd>${esc(d.n)}</dd>` : ""}${d.r ? `<dt>Reference</dt><dd>${esc(d.r)}</dd>` : ""}
      ${d.dep ? `<dt>Department</dt><dd>${esc(d.dep)}</dd>` : ""}${d.v ? `<dt>Version</dt><dd>${esc(d.v)}</dd>` : ""}
    </dl>
    <div class="sb-callout sb-callout--info docnote"><p><strong>Latest in the indexed corpus:</strong> ${latest ? `${title(latest)} (${date(latest)})` : "this document"}.
      ${esc(g.disclaimer)}</p></div>
    ${versions.length > 1 ? `<h3 class="dochead">Versions in the corpus</h3><ol class="doclist">${versions.map((v) => `<li${v === docId ? ' aria-current="true"' : ""}>${title(v)} <span class="meta">(${date(v)})</span></li>`).join("")}</ol>` : ""}
    ${changedBy.length ? `<h3 class="dochead">Changed by</h3><ul class="doclist">${changedBy.map(item).join("")}</ul>` : ""}
    ${changes.length ? `<h3 class="dochead">Changes</h3><ul class="doclist">${changes.map(item).join("")}</ul>` : ""}
    ${refs.length || citedBy.length ? `<h3 class="dochead">Related</h3><ul class="doclist">${[...refs.slice(0, 6).map((e) => ({ ...e, type: "refers to" })), ...citedBy.slice(0, 6).map((e) => ({ ...e, type: "cited by" }))].map(item).join("")}</ul>` : ""}
    <p class="meta">Relations are extracted from the text with patterns (circular / reference numbers, "amended vide", "in supersession of", ...). References that could not be matched to one document are left out as unknown, never guessed.</p>`;
}
