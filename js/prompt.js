// Same prompt as rbirag.generation.prompts (the system prompt and refusal text are
// exported into the index manifest so the two never drift apart).

export function formatSource(n, c) {
  const parts = [c.title || "Untitled"];
  for (const key of ["circular_number", "date", "department"]) if (c[key]) parts.push(String(c[key]));
  if (c.section) parts.push(`section: ${c.section}`);
  return `[${n}] ${parts.join(" | ")}\n${c.text.trim()}`;
}

export function buildUserPrompt(question, chunks) {
  const context = chunks.length
    ? chunks.map((c, i) => formatSource(i + 1, c)).join("\n\n")
    : "(no sources were retrieved)";
  return `Sources:\n\n${context}\n\nQuestion: ${question.trim()}`;
}

export function citedIds(answer) {
  const ids = [];
  // 1-3 digit ids only, as in rbirag.generation.citations ("[1949]" is not a citation)
  for (const m of answer.matchAll(/\[\s*(\d{1,3}(?:\s*(?:,|-)\s*\d{1,3})*)\s*\]/g)) {
    for (const part of m[1].split(/\s*,\s*/)) {
      const r = part.match(/^(\d+)\s*-\s*(\d+)$/);
      if (r) for (let i = +r[1]; i <= +r[2] && i - +r[1] <= 20; i++) ids.push(i);
      else if (/^\d+$/.test(part)) ids.push(+part);
    }
  }
  return [...new Set(ids)];
}
