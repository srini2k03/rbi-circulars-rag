"""Prompt templates: numbered source chunks and a citation-first answer contract."""

from __future__ import annotations

from collections.abc import Sequence

from rbirag.chunking.base import Chunk
from rbirag.generation.llm import Message

REFUSAL_TEXT = "I don't know based on the provided RBI notifications."

SYSTEM_PROMPT = f"""You answer questions about Reserve Bank of India (RBI) notifications, \
circulars and directions, using only the numbered sources supplied with each question.

How to answer:
- Use only facts stated in the sources. Do not use outside knowledge, even if you are \
confident about it.
- After every sentence that states a fact, cite the source(s) it comes from with square \
brackets, e.g. "Banks must update KYC every two years for high-risk customers [2]." \
Several sources look like [1][3].
- Cite only source numbers that exist in this prompt.
- Keep the answer short and specific: lead with the direct answer, include the relevant \
numbers, dates and circular references, and add brief qualifying conditions only if the \
sources state them.
- If the sources do not contain the answer, reply with exactly this sentence and nothing \
else: {REFUSAL_TEXT}"""


def format_source(n: int, chunk: Chunk) -> str:
    md = chunk.metadata
    header_parts = [str(md.get("title") or "Untitled")]
    for key in ("circular_number", "date", "department"):
        value = md.get(key)
        if value:
            header_parts.append(str(value))
    if chunk.section:
        header_parts.append(f"section: {chunk.section}")
    return f"[{n}] {' | '.join(header_parts)}\n{chunk.text.strip()}"


def build_context(chunks: Sequence[Chunk]) -> str:
    return "\n\n".join(format_source(i, c) for i, c in enumerate(chunks, start=1))


def build_user_prompt(question: str, chunks: Sequence[Chunk]) -> str:
    context = build_context(chunks) if chunks else "(no sources were retrieved)"
    return f"Sources:\n\n{context}\n\nQuestion: {question.strip()}"


def build_messages(question: str, chunks: Sequence[Chunk]) -> list[Message]:
    return [Message(role="user", content=build_user_prompt(question, chunks))]
