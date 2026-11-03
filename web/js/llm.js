// Optional answer generation with a visitor-supplied key. The key is sent only to the
// chosen provider's API, directly from this browser.

export const PROVIDERS = {
  groq: { label: "Groq", keyUrl: "https://console.groq.com/keys" },
  gemini: { label: "Google Gemini", keyUrl: "https://aistudio.google.com/apikey" },
};

function stripReasoning(text) {
  return text.replace(/<think>[\s\S]*?(<\/think>|$)\s*/gi, "").trim();
}

export async function generate({ provider, key, model, system, user, fetchImpl = fetch }) {
  if (provider === "groq") {
    const resp = await fetchImpl("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
      body: JSON.stringify({
        model,
        max_tokens: 1024,
        messages: [{ role: "system", content: system }, { role: "user", content: user }],
      }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(`Groq HTTP ${resp.status}: ${data?.error?.message || ""}`);
    return { text: stripReasoning(data.choices?.[0]?.message?.content || ""), usage: data.usage || null, model: data.model || model };
  }
  if (provider === "gemini") {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`;
    const resp = await fetchImpl(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-goog-api-key": key },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: system }] },
        contents: [{ role: "user", parts: [{ text: user }] }],
        generationConfig: { maxOutputTokens: 1024 },
      }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(`Gemini HTTP ${resp.status}: ${data?.error?.message || ""}`);
    const parts = data.candidates?.[0]?.content?.parts || [];
    return { text: stripReasoning(parts.map((p) => p.text || "").join("")), usage: data.usageMetadata || null, model };
  }
  throw new Error(`unknown provider ${provider}`);
}
