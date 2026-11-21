"""Keep the planner's conversation inside a token budget.

Each search or fetch result stays in the history, so a clause that needs several searches
can outgrow a small model's request limit (Groq's free tier allows about 7-8K input
tokens per minute, so one oversized request is rejected outright). Before each planner
call the oldest tool results are shortened, keeping what the planner needs to carry on
(which notifications it saw, their titles and entities, any comparison) and dropping
snippets and passages it can fetch again.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from rbirag.compliance.llm import ChatMessage
from rbirag.generation.llm import approx_tokens

SHORTENED = "[shortened to save context; call the tool again if you need the text] "


def message_tokens(msg: ChatMessage) -> int:
    n = approx_tokens(msg.content or "") + 4
    for call in msg.tool_calls:
        n += approx_tokens(call.name) + approx_tokens(call.arguments) + 4
    return n


def history_tokens(messages: Sequence[ChatMessage], tools: Sequence[dict[str, Any]] = ()) -> int:
    extra = approx_tokens(json.dumps(list(tools))) if tools else 0
    return sum(message_tokens(m) for m in messages) + extra


def shorten_result(content: str) -> str:
    """A short stand-in for one tool result (compare_clause results are kept)."""
    if content.startswith(SHORTENED) or content.startswith("ERROR:"):
        return content
    try:
        data = json.loads(content.removesuffix(" (repeated call)"))
    except json.JSONDecodeError:
        return SHORTENED + content[:300]
    if not isinstance(data, dict):
        return SHORTENED + content[:300]
    if "hits" in data:  # search_regulations
        hits = [
            {k: h.get(k) for k in ("doc_id", "title", "date", "entity") if h.get(k)}
            for h in data.get("hits", [])
            if isinstance(h, dict)
        ]
        return SHORTENED + json.dumps({"hits": hits}, ensure_ascii=False, separators=(",", ":"))
    if "passages" in data:  # fetch_notification
        paras = [p.get("para") for p in data.get("passages", []) if isinstance(p, dict)]
        kept = {k: data.get(k) for k in ("doc_id", "title", "date", "entity", "replaced_by")}
        kept["passages_read"] = [p for p in paras if p]
        return SHORTENED + json.dumps(kept, ensure_ascii=False, separators=(",", ":"))
    return content  # compare_clause: small, and the planner should adopt it


def fit_history(
    messages: list[ChatMessage],
    budget: int,
    tools: Sequence[dict[str, Any]] = (),
) -> int:
    """Shorten tool results, oldest first, until the history fits ``budget`` tokens.

    Mutates ``messages`` in place and returns how many results were shortened. The system
    prompt, the clause and the assistant turns are never changed.
    """
    shortened = 0
    for i, msg in enumerate(messages):
        if history_tokens(messages, tools) <= budget:
            break
        if msg.role != "tool" or not msg.content:
            continue
        short = shorten_result(msg.content)
        if short != msg.content:
            messages[i] = msg.model_copy(update={"content": short})
            shortened += 1
    return shortened
