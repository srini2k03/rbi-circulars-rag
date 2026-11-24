// Evaluation-tab cards for the retrieval benchmark, the judge-vs-NLI study, citation
// verification, query understanding and the relation extraction check. Every number is read
// from results/ (each card is skipped when its file is not there yet).

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const f = (v, d = 3) => (v == null || Number.isNaN(+v) ? "-" : (+v).toFixed(d));
const pct = (v) => (v == null ? "-" : `${(100 * v).toFixed(1)}%`);
const signed = (v, d = 3) => (v == null ? "-" : `${v >= 0 ? "+" : ""}${(+v).toFixed(d)}`);

async function json(url) {
  try {
    const r = await fetch(url);
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}

const CATS = [
  ["top1", "gold passage ranked 1st", "var(--live)"],
  ["in_context", "in ranks 2-5", "var(--series-1)"],
  ["ranked_low", "retrieved, ranked below 5", "var(--pending)"],
  ["right_doc_wrong_passage", "right document, other passage", "var(--series-2)"],
  ["related_doc", "related document only", "var(--faint)"],
  ["not_retrieved", "not retrieved", "var(--danger)"],
];

function benchmarkCard(b) {
  if (!b?.rows?.length) return "";
  const rows = b.rows.map((r) => {
    const c = r.vs_bm25;
    const total = Object.values(r.categories || {}).reduce((a, x) => a + x, 0) || 1;
    const bar = CATS.map(([k, label, color]) => {
      const n = r.categories?.[k] || 0;
      return n ? `<span style="width:${((100 * n) / total).toFixed(2)}%;background:${color}" title="${esc(label)}: ${n}"></span>` : "";
    }).join("");
    const oracle = r.family.includes("oracle");
    return `<tr${r.strategy === b.recommended ? ` class="is-best"` : ""}>
      <td class="cfg">${esc(r.family)}${oracle ? ` <span class="sb-tag" title="Uses the gold document's metadata: an upper bound, not a deployable system">oracle</span>` : ""}<span class="ci">${esc(r.label)}</span></td>
      <td class="num">${f(r["dev_chunk_ndcg@10"])}</td>
      <td class="num">${f(r["chunk_ndcg@10"])}<span class="ci">[${f(r["chunk_ndcg@10_ci"]?.[0])}, ${f(r["chunk_ndcg@10_ci"]?.[1])}]</span></td>
      <td class="num">${c ? `${signed(c.diff)}<span class="ci">[${signed(c.low)}, ${signed(c.high)}] p=${f(c.p_holm, 3)}</span>` : "baseline"}</td>
      <td class="num">${f(r["chunk_recall@5"])}</td><td class="num">${f(r["doc_recall@5"])}</td>
      <td class="num">${f(r.coverage)}</td>
      <td><span class="catbar" role="img" aria-label="${esc(CATS.map(([k, l]) => `${l}: ${r.categories?.[k] || 0}`).join(", "))}">${bar}</span></td></tr>`;
  }).join("");
  const legend = CATS.map(([, label, color]) => `<span class="legend__item"><span class="legend__swatch" style="background:${color}"></span>${esc(label)}</span>`).join("");
  const abst = Object.entries(b.abstention || {}).filter(([, v]) => v && typeof v === "object")
    .map(([k, v]) => `<li><strong>${esc(k)}</strong>: AUROC ${f(v.auroc_test)} on test; at the dev-chosen threshold, refusal precision ${f(v.test?.refusal_precision, 2)}, recall ${f(v.test?.refusal_recall, 2)}</li>`).join("");
  const live = (b.live_timing || []).map((t) => `<tr><td class="cfg">${esc(t.strategy)}</td><td class="num">${f(t.total_ms_p50, 1)}</td><td class="num">${f(t.total_ms_p95, 1)}</td><td class="num">${esc(t.identical_top10_vs_cached)}</td></tr>`).join("");
  return `<figure class="ev-card sb-card">
    <figcaption class="ev-card__cap"><p class="sb-card__title">Retrieval strategies: tuned on dev, reported on test</p>
      <p class="sb-card__sub">${esc(b.n_test_answerable)} answerable test questions; every weight, depth and threshold was chosen on ${esc(b.n_dev_answerable)} dev questions (${esc(b.tuning?.length)} configurations tried).
      Δ is the paired difference in chunk nDCG@10 against plain BM25 with its bootstrap CI and Holm-adjusted randomisation-test p. Index: ${esc(b.primary)}.</p></figcaption>
    <div class="sb-table-wrap"><table class="sb-table rank-table bench-table">
      <thead><tr><th>strategy (dev-selected)</th><th class="num">dev nDCG@10</th><th class="num">test nDCG@10</th><th class="num">Δ vs BM25</th><th class="num">chunk R@5</th><th class="num">doc R@5</th><th class="num">coverage@5</th><th>where the gold passage ended up</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
    <p class="legend">${legend}</p>
    ${abst ? `<p class="sb-card__sub">Can retrieval tell unanswerable questions apart (${esc(b.abstention.n_test_unanswerable)} on test, only ${esc(b.abstention.n_dev_unanswerable)} on dev to set thresholds)?</p><ul class="ev-list">${abst}</ul>` : ""}
    ${live ? `<div class="sb-table-wrap"><table class="sb-table"><thead><tr><th>live run (runner CPU)</th><th class="num">p50 ms</th><th class="num">p95 ms</th><th class="num">top-10 = cached run</th></tr></thead><tbody>${live}</tbody></table></div>` : ""}
    <p class="sb-card__foot">Source: results/benchmark/strategies.json (generated ${esc(b.generated_at)}, commit <code>${esc((b.commit || "").slice(0, 7))}</code>)</p>
  </figure>`;
}

function nliCard(n) {
  if (!n?.whole_chunk) return "";
  const names = { whole_chunk: "whole passage (original)", windows_cited: "3-sentence windows of the cited passages", windows_any: "windows of any retrieved passage", claims_windows: "clause-level claims vs windows" };
  const rows = Object.entries(names).map(([k, label]) => `<tr><td class="cfg">${esc(label)}</td><td class="num">${f(n[k]?.mean?.mean)}<span class="ci">[${f(n[k]?.mean?.low)}, ${f(n[k]?.mean?.high)}]</span></td><td class="num">${f(n[k]?.pearson_r_vs_judge, 2)}</td><td class="num">${f(n[k]?.mean_abs_diff_vs_judge)}</td></tr>`).join("");
  return `<figure class="ev-card sb-card">
    <figcaption class="ev-card__cap"><p class="sb-card__title">Why the LLM judge and NLI disagree on faithfulness</p>
      <p class="sb-card__sub">${esc(n.answers)} committed answers re-scored with ${esc(n.nli_model)} under four protocols; judge mean ${f(n.judge_mean?.mean)}.
      ${pct(n.premises_truncated_share)} of the passages exceed the model's 512-token input; ${pct(n.uncited_sentence_share)} of answer sentences cite no valid source;
      ${esc(n.sentences_rescued_by_windows)} sentences are entailed by a window but not by the whole passage (${esc(n.rescued_beyond_512_cutoff)} of them beyond the truncation point).</p></figcaption>
    <div class="sb-table-wrap"><table class="sb-table"><thead><tr><th>NLI protocol</th><th class="num">faithfulness</th><th class="num">r vs judge</th><th class="num">|Δ| vs judge</th></tr></thead><tbody>${rows}</tbody></table></div>
    <p class="sb-card__foot">Source: results/analysis/nli_study.json</p></figure>`;
}

function verifyCard(v, label) {
  const a = v?.all;
  if (!a) return "";
  const total = a.answers || 1;
  const parts = [["supported", "var(--live)"], ["partially_supported", "var(--pending)"], ["unsupported", "var(--danger)"], ["no_claims", "var(--faint)"]];
  const bar = parts.map(([k, c]) => (a.verdicts?.[k] ? `<span style="width:${((100 * a.verdicts[k]) / total).toFixed(2)}%;background:${c}" title="${k}: ${a.verdicts[k]}"></span>` : "")).join("");
  return `<figure class="ev-card sb-card">
    <figcaption class="ev-card__cap"><p class="sb-card__title">Citation verification of the committed answers (${esc(label)})</p>
      <p class="sb-card__sub">${esc(a.answers)} answers; claims are checked against windows of their cited passages (threshold ${esc(v.threshold)}). A check of the answers, not ground truth.</p></figcaption>
    <p><span class="catbar catbar--wide" role="img" aria-label="${esc(parts.map(([k]) => `${k}: ${a.verdicts?.[k] || 0}`).join(", "))}">${bar}</span></p>
    <ul class="ev-list"><li>supported ${esc(a.verdicts?.supported)}, partially supported ${esc(a.verdicts?.partially_supported)}, unsupported ${esc(a.verdicts?.unsupported)}</li>
      <li>unsupported claims: ${pct(a.unsupported_claim_rate)}; cited passages supporting none of the claims citing them: ${pct(a.irrelevant_citation_rate)}</li>
      <li>answers citing a source that does not exist: ${esc(a.answers_with_invalid_citation)}; naming a circular number found in no source: ${esc(a.answers_with_unknown_circular_numbers)}</li>
      <li>of the ${esc(a.judge_fully_faithful)} answers the LLM judge rated fully faithful, ${esc(a.of_which_verifier_supported)} pass this check</li></ul>
    <p class="sb-card__foot">Source: results/analysis/citation_verification-${esc(label)}.json</p></figure>`;
}

function intelCard(p, s) {
  if (!p?.all_relations || !s) return "";
  const types = Object.entries(p).filter(([k, v]) => v && v.both && !["all_relations", "population_weighted_precision"].includes(k))
    .map(([k, v]) => `<tr><td class="cfg">${esc(k.replace(/_/g, " "))}</td><td class="num">${esc(v.both.n)}</td><td class="num">${f(v.both.precision, 2)}<span class="ci">[${f(v.both.ci95?.[0], 2)}, ${f(v.both.ci95?.[1], 2)}]</span></td></tr>`).join("");
  return `<figure class="ev-card sb-card">
    <figcaption class="ev-card__cap"><p class="sb-card__title">Document intelligence: relations between notifications</p>
      <p class="sb-card__sub">${esc(s.references)} references found in ${esc(s.documents)} documents; ${esc(s.references_resolved)} resolve to one indexed document, the rest stay <em>unknown</em> (outside the corpus or ambiguous).
      ${esc(s.edges)} distinct relations. Precision on a seeded sample, ${esc(p.label)}: ${f(p.all_relations.precision, 3)} (n=${esc(p.all_relations.n)}, 95% CI ${f(p.all_relations.ci95?.[0], 2)}-${f(p.all_relations.ci95?.[1], 2)}). Recall was not measured.</p></figcaption>
    <div class="sb-table-wrap"><table class="sb-table"><thead><tr><th>type</th><th class="num">checked</th><th class="num">precision (target + type)</th></tr></thead><tbody>${types}</tbody></table></div>
    <p class="sb-card__foot">Source: results/intel/precision.json, results/intel/summary.json</p></figure>`;
}

function queryCard(q) {
  const fu = q?.follow_up;
  if (!fu) return "";
  return `<figure class="ev-card sb-card">
    <figcaption class="ev-card__cap"><p class="sb-card__title">Query understanding</p>
      <p class="sb-card__sub">Behavioural checks: the follow-ups, ambiguous, multi-part and out-of-scope inputs are AI-authored (data/eval/query_understanding_ai.jsonl), not real user queries.</p></figcaption>
    <div class="sb-table-wrap"><table class="sb-table"><thead><tr><th>follow-up turn (n=${esc(fu.n)}), BM25</th><th class="num">doc R@5</th><th class="num">chunk nDCG@10</th></tr></thead><tbody>
      <tr><td class="cfg">as asked (elliptical)</td><td class="num">${f(fu.alone?.["doc_recall@5"], 2)}</td><td class="num">${f(fu.alone?.["chunk_ndcg@10"])}</td></tr>
      <tr class="is-best"><td class="cfg">rewritten with the previous question</td><td class="num">${f(fu.rewritten?.["doc_recall@5"], 2)}</td><td class="num">${f(fu.rewritten?.["chunk_ndcg@10"])}</td></tr>
      <tr><td class="cfg">original full question (upper bound)</td><td class="num">${f(fu.full?.["doc_recall@5"], 2)}</td><td class="num">${f(fu.full?.["chunk_ndcg@10"])}</td></tr></tbody></table></div>
    <ul class="ev-list"><li>unanswerable test questions refused as out of scope: ${esc(q.test_unanswerable?.["out_of_scope (correct abstain)"]?.flagged)}/${esc(q.test_unanswerable?.["out_of_scope (correct abstain)"]?.n)}; answerable ones wrongly flagged: ${esc(q.test_answerable?.["out_of_scope (false abstain)"]?.flagged)}/${esc(q.test_answerable?.["out_of_scope (false abstain)"]?.n)}</li>
      <li>AI-authored inputs flagged: out of scope ${esc(q.ai_out_of_scope?.flagged)}/${esc(q.ai_out_of_scope?.n)}, ambiguous ${esc(q.ai_ambiguous?.flagged)}/${esc(q.ai_ambiguous?.n)}, multi-part ${esc(q.ai_multi_part?.flagged)}/${esc(q.ai_multi_part?.n)}; follow-ups detected ${esc(fu.detected_as_follow_up)}/${esc(fu.n)}</li></ul>
    <p class="sb-card__foot">Source: results/analysis/query_understanding.json</p></figure>`;
}

export async function analysisCards(base) {
  const [bench, nli, verLex, verNli, prec, summ, qu] = await Promise.all([
    json(`${base}/benchmark/strategies.json`), json(`${base}/analysis/nli_study.json`),
    json(`${base}/analysis/citation_verification-lexical.json`), json(`${base}/analysis/citation_verification-nli.json`),
    json(`${base}/intel/precision.json`), json(`${base}/intel/summary.json`), json(`${base}/analysis/query_understanding.json`),
  ]);
  const cards = [benchmarkCard(bench), nliCard(nli), verifyCard(verNli, "nli"), verifyCard(verLex, "lexical"), queryCard(qu), intelCard(prec, summ)].filter(Boolean);
  if (!cards.length) return "";
  return `<section class="eval-analysis" aria-labelledby="analysis-title">
    <p class="sb-kicker"><span class="sb-kicker__index">03</span><span class="sb-kicker__rule"></span><span id="analysis-title">Retrieval strategies, verification and document intelligence</span></p>
    ${cards.join("\n")}</section>`;
}
